"""adaptive_simulator.py - the tick-by-tick, stateful simulation engine; owns all mutable state, calls into stateless controller_core.py for decisions."""

from collections import Counter, deque

import numpy as np

from simulation.movement_definitions import (
    MOVEMENT_IDS,
    TRAVEL_DIRECTIONS,
    DIRECTION_MOVEMENTS,
    PHASES,
    PHASE_NAMES,
    PHYSICAL_APPROACH_SIDE,
)

from simulation.controller_core import (
    CITY,
    INTERSECTION_ID,
    SIMULATION_HOUR,
    WEEKEND,
    MONTH,
    SIMULATION_DURATION,
    MIN_GREEN,
    MAX_GREEN,
    YELLOW_DURATION,
    SERVICE_RATE,
    MAX_STARVATION_TIME,
    ENTRY_HEADING_LETTER,
    load_turning_proportions,
    load_arrival_rates,
    load_ml_model,
    load_ir_retriever,
    build_ml_predictions,
    build_ir_predictions,
    derive_exit_heading,
    parse_movement_id,
    get_average_wait,
    calculate_movement_priority,
    calculate_movement_priority_breakdown,
    calculate_phase_priority,
    should_switch_phase,
)

from rag.traffic_rag import generate_approach_analysis

RANDOM_SEED = 42

DEFAULT_STARTING_PHASE = "NS_THROUGH"

# One simulated tick = one second, so 3600 ticks = one simulated
# hour. Used to decide when to refresh ML/IR evidence.
SECONDS_PER_SIMULATED_HOUR = 3600

# Order movement types are sampled in for each arriving vehicle.
# Fixed and documented so the same seed always produces the same
# sequence of rng calls (required for reproducibility).
MOVEMENT_TYPE_SAMPLE_ORDER = [
    "STRAIGHT",
    "LEFT",
    "RIGHT",
]

# Dashboard/simulation lane identifiers.  These do not create controller
# movements: LEFT and RIGHT retain one lane, while each existing STRAIGHT
# movement owns two independently queued lanes.  The controller continues to
# receive the original 12 aggregate movement queues.
LANE_SUFFIXES = {
    "LEFT": ("LEFT",),
    "STRAIGHT": ("STRAIGHT_1", "STRAIGHT_2"),
    "RIGHT": ("RIGHT",),
}


def lane_ids_for_movement(movement_id):
    """Return visualization lane IDs for one existing controller movement."""

    direction, movement_type = parse_movement_id(movement_id)

    return [
        f"{direction}_{suffix}"
        for suffix in LANE_SUFFIXES[movement_type]
    ]


class Vehicle:
    """
    One simulated vehicle. Deliberately a plain, small object - no
    behavior, just the facts about one vehicle, so it is easy to
    inspect/explain in a viva.

    Waiting time is NOT stored - it is always derived as
    (current_tick - arrival_tick), computed only when needed, per
    the "no redundant mutable state" requirement.
    """

    __slots__ = (
        "vehicle_id",
        "direction",
        "movement_type",
        "movement_id",
        "entry_heading",
        "exit_heading",
        "arrival_tick",
    )

    def __init__(
        self,
        vehicle_id,
        direction,
        movement_type,
        movement_id,
        entry_heading,
        exit_heading,
        arrival_tick,
    ):
        self.vehicle_id = vehicle_id
        self.direction = direction
        self.movement_type = movement_type
        self.movement_id = movement_id
        self.entry_heading = entry_heading
        self.exit_heading = exit_heading
        self.arrival_tick = arrival_tick

    def waiting_time(self, current_tick):
        return current_tick - self.arrival_tick


class AdaptiveTrafficSimulation:
    """
    Movement-aware, tick-by-tick adaptive traffic simulation.

    Public methods: reset(), step(), get_state().
    Everything else is an internal implementation detail.
    """

    def __init__(
        self,
        hour=SIMULATION_HOUR,
        weekend=WEEKEND,
        month=MONTH,
        random_seed=RANDOM_SEED,
        arrival_rate_overrides=None,
        max_starvation_time=MAX_STARVATION_TIME,
        generate_rag_explanations=True,
    ):
        """
        arrival_rate_overrides: optional {direction: rate} dict that
        overrides the real historical rate for specific directions.
        Used ONLY for constructing deterministic test/stress
        scenarios (see tests/test_starvation_stress.py) - the
        default (None) always uses the real, historically-derived
        rates from load_arrival_rates().

        max_starvation_time: hard fairness safeguard threshold
        (ticks/seconds). None (the default) disables it, leaving
        only the weighted formula's starvation component active.
        Passed straight through to
        controller_core.should_switch_phase() every tick.

        generate_rag_explanations: whether to call the Gemini API
        (real network calls, 4 per hour-boundary refresh) to
        populate rag_explanations. Defaults to True for the live
        dashboard. Tests that don't exercise RAG should pass False -
        otherwise every simulator constructed in a test run makes 4
        real, billed API calls unrelated to what's being tested.
        """

        self.hour = int(hour)
        self.weekend = int(weekend)
        self.month = int(month)
        self.random_seed = int(random_seed)
        self.max_starvation_time = max_starvation_time
        self.generate_rag_explanations = (
            generate_rag_explanations
        )

        # Loaded once - these don't change tick to tick, so there's
        # no reason to reload them on every step.
        self.model_bundle = load_ml_model()
        self.ir_retriever = load_ir_retriever()

        self.turning_proportions = load_turning_proportions()

        # Cached as the ORIGINAL hour's predictions - reset() always
        # restores these, even if a mid-run hourly refresh had
        # replaced self.ml_predictions/ir_predictions with a later
        # hour's values before reset() was called.
        self._initial_ml_predictions = build_ml_predictions(
            model_bundle=self.model_bundle,
            hour=self.hour,
            weekend=self.weekend,
            month=self.month,
        )

        self._initial_ir_predictions = build_ir_predictions(
            retriever=self.ir_retriever,
            hour=self.hour,
            weekend=self.weekend,
            month=self.month,
            ml_predictions=self._initial_ml_predictions,
            top_k=5,
        )

        self._initial_rag_explanations = (
            self._build_rag_explanations(
                self._initial_ml_predictions,
                self._initial_ir_predictions,
            )
        )

        self.arrival_rates = load_arrival_rates(
            hour=self.hour,
            weekend=self.weekend,
        )

        if arrival_rate_overrides:

            self.arrival_rates = dict(self.arrival_rates)

            self.arrival_rates.update(
                arrival_rate_overrides
            )

        self.reset()

    def reset(self):

        self.tick = 0

        self.movement_queues = {
            movement_id: deque()
            for movement_id in MOVEMENT_IDS
        }

        # The aggregate movement queues above remain the controller's public
        # compatibility surface.  Straight lane queues store the individual
        # lane placement; left/right lanes alias their sole movement queue.
        self.lane_queues = {}

        for movement_id in MOVEMENT_IDS:

            lane_ids = lane_ids_for_movement(movement_id)

            if len(lane_ids) == 1:
                self.lane_queues[lane_ids[0]] = (
                    self.movement_queues[movement_id]
                )
            else:
                for lane_id in lane_ids:
                    self.lane_queues[lane_id] = deque()

        self.movement_total_arrivals = {
            movement_id: 0
            for movement_id in MOVEMENT_IDS
        }

        self.movement_total_departures = {
            movement_id: 0
            for movement_id in MOVEMENT_IDS
        }

        self.completed_waits = {
            movement_id: []
            for movement_id in MOVEMENT_IDS
        }

        self.movement_max_queues = {
            movement_id: 0
            for movement_id in MOVEMENT_IDS
        }

        self.movement_starvation = {
            movement_id: 0
            for movement_id in MOVEMENT_IDS
        }

        self.current_phase = DEFAULT_STARTING_PHASE
        self.target_phase = None
        self.signal_state = "GREEN"
        self.green_elapsed = 0
        self.yellow_elapsed = 0

        self.adaptive_switches = 0
        self.switch_reasons = Counter()
        self.starvation_override_switches = 0

        self.simulated_hour = self.hour
        self.ml_predictions = dict(self._initial_ml_predictions)
        self.ir_predictions = dict(self._initial_ir_predictions)
        self.rag_explanations = dict(
            self._initial_rag_explanations
        )
        self.ml_ir_refresh_history = [
            {"tick": 0, "hour": self.hour}
        ]

        self.movement_priorities = {
            movement_id: 0.0
            for movement_id in MOVEMENT_IDS
        }

        self.phase_priorities = {
            phase_name: 0.0
            for phase_name in PHASE_NAMES
        }

        self.last_decision = {
            "action": "INITIALIZED",
            "reason": "simulation_initialized",
            "ending_phase": None,
            "target_phase": DEFAULT_STARTING_PHASE,
            "tick": 0,
        }

        self.finished = False

        self._next_vehicle_id = 1

        self.rng = np.random.default_rng(
            self.random_seed
        )

        return self.get_state()

    def _refresh_ml_ir_if_hour_changed(self, tick):
        """
        Refreshes ML/IR/RAG evidence when the simulated hour
        advances (every SECONDS_PER_SIMULATED_HOUR ticks). Does
        NOT retrain the model or rebuild the IR index - it just
        calls the same build_ml_predictions()/build_ir_predictions()/
        _build_rag_explanations() again with the new hour, exactly
        as __init__ did for the starting hour. Wraps at 24 (hour 23
        -> hour 0).

        For a normal 300-tick or even a 3000-tick run this never
        fires (both are under 3600 ticks) - documented explicitly,
        not just assumed, and confirmed by
        tests/test_starvation_safeguard.py. The Gemini calls this
        triggers are therefore a one-time cost at
        __init__ / hour-change for any realistic run length, not a
        recurring one during Play.
        """

        elapsed_hours = (
            tick - 1
        ) // SECONDS_PER_SIMULATED_HOUR

        new_simulated_hour = (
            self.hour + elapsed_hours
        ) % 24

        if new_simulated_hour == self.simulated_hour:
            return

        self.simulated_hour = new_simulated_hour

        self.ml_predictions = build_ml_predictions(
            model_bundle=self.model_bundle,
            hour=self.simulated_hour,
            weekend=self.weekend,
            month=self.month,
        )

        self.ir_predictions = build_ir_predictions(
            retriever=self.ir_retriever,
            hour=self.simulated_hour,
            weekend=self.weekend,
            month=self.month,
            ml_predictions=self.ml_predictions,
            top_k=5,
        )

        self.rag_explanations = self._build_rag_explanations(
            self.ml_predictions, self.ir_predictions
        )

        self.ml_ir_refresh_history.append(
            {"tick": tick, "hour": self.simulated_hour}
        )

    def _sample_movement_type(self, direction):
        """
        Sample one movement type (STRAIGHT/LEFT/RIGHT) for a new
        vehicle in `direction`, using the real historical turning
        proportions loaded from turning_proportions.csv - never
        hardcoded here.
        """

        probabilities = [
            self.turning_proportions[
                f"{direction}_{movement_type}"
            ]
            for movement_type in MOVEMENT_TYPE_SAMPLE_ORDER
        ]

        # Guard against floating point rounding pushing the sum
        # very slightly away from 1.0 (rng.choice requires exact).
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()

        return self.rng.choice(
            MOVEMENT_TYPE_SAMPLE_ORDER,
            p=probabilities,
        )

    def _generate_arrivals(self, tick):

        for direction in TRAVEL_DIRECTIONS:

            arrival_count = int(
                self.rng.poisson(
                    self.arrival_rates[direction]
                )
            )

            entry_heading = ENTRY_HEADING_LETTER[
                direction
            ]

            for _ in range(arrival_count):

                movement_type = (
                    self._sample_movement_type(
                        direction
                    )
                )

                movement_id = (
                    f"{direction}_{movement_type}"
                )

                exit_heading = derive_exit_heading(
                    entry_heading,
                    movement_type,
                )

                vehicle = Vehicle(
                    vehicle_id=self._next_vehicle_id,
                    direction=direction,
                    movement_type=movement_type,
                    movement_id=movement_id,
                    entry_heading=entry_heading,
                    exit_heading=exit_heading,
                    arrival_tick=tick,
                )

                self._next_vehicle_id += 1

                if movement_type == "STRAIGHT":

                    straight_lane_ids = lane_ids_for_movement(
                        movement_id
                    )

                    first_lane, second_lane = (
                        self.lane_queues[lane_id]
                        for lane_id in straight_lane_ids
                    )

                    if len(first_lane) < len(second_lane):
                        selected_lane = first_lane
                    elif len(second_lane) < len(first_lane):
                        selected_lane = second_lane
                    else:
                        # A tie is a genuine driver choice.  Use the
                        # simulator RNG so equal-length behavior remains
                        # reproducible for a fixed random seed.
                        selected_lane = (
                            first_lane
                            if self.rng.integers(0, 2) == 0
                            else second_lane
                        )

                    selected_lane.append(vehicle)

                self.movement_queues[movement_id].append(
                    vehicle
                )

                self.movement_total_arrivals[
                    movement_id
                ] += 1

    def _serve_green_phase(self, tick):
        """
        Only movements in the currently active phase, and only
        while signal_state == "GREEN", may depart. YELLOW serves
        zero vehicles - deliberate, matching a real clearance
        interval where no new vehicles enter the intersection.
        """

        if self.signal_state != "GREEN":
            return

        service_capacity = int(SERVICE_RATE)

        for movement_id in PHASES[self.current_phase]:

            for lane_id in lane_ids_for_movement(movement_id):

                lane_queue = self.lane_queues[lane_id]

                for _ in range(service_capacity):

                    if not lane_queue:
                        break

                    vehicle = lane_queue.popleft()

                    # Straight vehicles exist in both the lane queue and the
                    # aggregate movement queue.  Remove the exact departed
                    # vehicle rather than assuming aggregate queue order,
                    # because the two straight lanes discharge independently.
                    if lane_queue is not self.movement_queues[movement_id]:
                        self.movement_queues[movement_id].remove(
                            vehicle
                        )

                    wait = vehicle.waiting_time(tick)

                    self.completed_waits[
                        movement_id
                    ].append(wait)

                    self.movement_total_departures[
                        movement_id
                    ] += 1

    def _update_max_queues(self):

        for movement_id in MOVEMENT_IDS:

            self.movement_max_queues[
                movement_id
            ] = max(
                self.movement_max_queues[movement_id],
                len(self.movement_queues[movement_id]),
            )

    def _update_starvation(self):

        active_movements = PHASES[self.current_phase]

        for movement_id in MOVEMENT_IDS:

            is_being_served_now = (
                self.signal_state == "GREEN"
                and movement_id in active_movements
            )

            if is_being_served_now:

                self.movement_starvation[
                    movement_id
                ] = 0

            elif self.movement_queues[movement_id]:

                self.movement_starvation[
                    movement_id
                ] += 1

            else:

                self.movement_starvation[
                    movement_id
                ] = 0

    def _arrival_tick_view(self):
        """
        controller_core.py's functions treat a "queue" as an
        iterable of arrival ticks (see get_average_wait,
        calculate_movement_priority). This simulator stores rich
        Vehicle objects instead, so this builds the plain
        arrival-tick view controller_core expects, once per tick.
        This keeps controller_core.py's decision logic completely
        unaware of the simulator's internal Vehicle representation -
        the adaptation happens here, in the simulator.
        """

        return {
            movement_id: [
                vehicle.arrival_tick
                for vehicle in self.movement_queues[
                    movement_id
                ]
            ]
            for movement_id in MOVEMENT_IDS
        }

    def _calculate_current_priorities(
        self,
        arrival_tick_queues,
    ):

        movement_priorities = {
            movement_id: float(
                calculate_movement_priority(
                    movement_id=movement_id,
                    movement_queues=arrival_tick_queues,
                    current_tick=self.tick,
                    ml_predictions=self.ml_predictions,
                    starvation=self.movement_starvation,
                    ir_predictions=self.ir_predictions,
                )
            )
            for movement_id in MOVEMENT_IDS
        }

        phase_priorities = {
            phase_name: float(
                calculate_phase_priority(
                    phase_name=phase_name,
                    movement_queues=arrival_tick_queues,
                    current_tick=self.tick,
                    ml_predictions=self.ml_predictions,
                    starvation=self.movement_starvation,
                    ir_predictions=self.ir_predictions,
                )
            )
            for phase_name in PHASE_NAMES
        }

        return movement_priorities, phase_priorities

    def _run_signal_control(
        self,
        tick,
        arrival_tick_queues,
    ):

        if self.signal_state == "GREEN":

            (
                should_switch,
                reason,
                target_phase,
                details,
            ) = should_switch_phase(
                current_phase=self.current_phase,
                green_elapsed=self.green_elapsed,
                movement_queues=arrival_tick_queues,
                current_tick=tick,
                ml_predictions=self.ml_predictions,
                starvation=self.movement_starvation,
                ir_predictions=self.ir_predictions,
                max_starvation_time=self.max_starvation_time,
            )

            # should_switch_phase() already computed every phase's
            # priority as `details` - reuse it directly instead of
            # recomputing. This intentionally overwrites any
            # separately-computed phase_priorities on every GREEN
            # tick rather than calculating them twice.
            self.phase_priorities = dict(details)

            if should_switch:

                self.target_phase = target_phase

                self.adaptive_switches += 1
                self.switch_reasons[reason] += 1

                if reason == "starvation_override":
                    self.starvation_override_switches += 1

                self.last_decision = {
                    "action": "SWITCH",
                    "reason": reason,
                    "ending_phase": self.current_phase,
                    "target_phase": target_phase,
                    "tick": tick,
                }

                self.signal_state = "YELLOW"
                self.yellow_elapsed = 0

            else:

                self.green_elapsed += 1

                self.last_decision = {
                    "action": "HOLD",
                    "reason": reason,
                    "ending_phase": None,
                    "target_phase": self.current_phase,
                    "tick": tick,
                }

        else:
            # YELLOW: no decision is made, just count elapsed time,
            # then activate the pre-selected target_phase once
            # YELLOW_DURATION is reached. Every phase change passes
            # through this state - there is no direct GREEN->GREEN
            # transition anywhere in this class.

            self.yellow_elapsed += 1

            self.last_decision = {
                "action": "TRANSITION",
                "reason": "yellow_clearance_interval",
                "ending_phase": self.current_phase,
                "target_phase": self.target_phase,
                "tick": tick,
            }

            if self.yellow_elapsed >= YELLOW_DURATION:

                previous_phase = self.current_phase

                self.current_phase = self.target_phase
                self.target_phase = None

                self.signal_state = "GREEN"
                self.green_elapsed = 0
                self.yellow_elapsed = 0

                self.last_decision = {
                    "action": "ACTIVATE",
                    "reason": "yellow_clearance_complete",
                    "ending_phase": previous_phase,
                    "target_phase": self.current_phase,
                    "tick": tick,
                }

    def step(self):

        if self.finished:
            return self.get_state()

        self.tick += 1
        tick = self.tick

        self._refresh_ml_ir_if_hour_changed(tick)

        self._generate_arrivals(tick)

        self._serve_green_phase(tick)

        self._update_max_queues()

        self._update_starvation()

        arrival_tick_queues = self._arrival_tick_view()

        (
            self.movement_priorities,
            self.phase_priorities,
        ) = self._calculate_current_priorities(
            arrival_tick_queues
        )

        self._run_signal_control(
            tick,
            arrival_tick_queues,
        )

        if self.tick >= SIMULATION_DURATION:
            self.finished = True

        return self.get_state()

    def run_ticks(self, number_of_ticks):

        number_of_ticks = max(1, int(number_of_ticks))

        state = self.get_state()

        for _ in range(number_of_ticks):

            if self.finished:
                break

            state = self.step()

        return state

    def _movement_signals(self):

        active_movements = PHASES[self.current_phase]

        signals = {}

        for movement_id in MOVEMENT_IDS:

            if movement_id in active_movements:
                signals[movement_id] = self.signal_state

            else:
                signals[movement_id] = "RED"

        return signals

    def _approach_signals(self, movement_signals):
        """
        An approach (NB/SB/EB/WB) has 3 movements that can be in
        DIFFERENT phases (e.g. NB_STRAIGHT/NB_RIGHT are in
        NS_THROUGH, NB_LEFT is in NS_LEFT), so a single approach
        doesn't always have one signal color. If all 3 agree, use
        it; otherwise report "MIXED" rather than picking one
        arbitrarily.
        """

        approach_signals = {}

        for direction in TRAVEL_DIRECTIONS:

            states = {
                movement_signals[movement_id]
                for movement_id in DIRECTION_MOVEMENTS[
                    direction
                ]
            }

            if len(states) == 1:
                approach_signals[direction] = states.pop()

            else:
                approach_signals[direction] = "MIXED"

        return approach_signals

    def _approach_ir_prediction_for_rag(
        self, direction, ir_predictions
    ):
        """
        rag/generator.py wants one IR severity + one set of
        retrieved cases per APPROACH, but build_ir_predictions()
        (like the rest of this movement-aware controller) only
        produces per-MOVEMENT IR evidence (12 keys, e.g.
        "NB_LEFT") - there is no "NB" key to hand it directly.
        Bridges that the same way _approach_live_congestion()
        already bridges movement-level congestion up to
        approach-level: mean severity across the direction's 3
        movements, plus the union of their retrieved historical
        cases so the explanation is grounded in all of them, not
        just one arbitrarily-picked movement.
        """

        movement_predictions = [
            ir_predictions[movement_id]
            for movement_id in DIRECTION_MOVEMENTS[direction]
        ]

        mean_severity = float(
            np.mean(
                [
                    prediction["severity"]
                    for prediction in movement_predictions
                ]
            )
        )

        combined_top_results = [
            result
            for prediction in movement_predictions
            for result in prediction["top_results"]
        ]

        return {
            "severity": mean_severity,
            "retrieved_cases": len(combined_top_results),
            "top_results": combined_top_results,
        }

    def _build_rag_explanations(
        self, ml_predictions, ir_predictions
    ):
        """
        Grounded Gemini explanation per travel direction, generated
        from the SAME ml_predictions/ir_predictions this refresh
        cycle already computed - no separate evidence is fetched.
        Mirrors rag/traffic_rag.py's own per-approach loop,
        including its per-approach try/except: a Gemini failure
        (missing key, network, quota) degrades to an error string
        for that one direction rather than crashing the simulator.

        Skipped entirely (no network call) when
        generate_rag_explanations=False was passed to __init__ -
        tests exercising unrelated simulator mechanics shouldn't pay
        for 4 real Gemini calls per instance.
        """

        if not self.generate_rag_explanations:

            return {
                direction: (
                    "RAG explanations disabled for this "
                    "simulator instance "
                    "(generate_rag_explanations=False)."
                )
                for direction in TRAVEL_DIRECTIONS
            }

        explanations = {}

        for direction in TRAVEL_DIRECTIONS:

            try:

                explanations[direction] = (
                    generate_approach_analysis(
                        approach=direction,
                        ml_prediction=ml_predictions[
                            direction
                        ],
                        ir_prediction=(
                            self._approach_ir_prediction_for_rag(
                                direction, ir_predictions
                            )
                        ),
                    )
                )

            except Exception as error:

                explanations[direction] = (
                    "Gemini explanation generation "
                    f"failed: {error}"
                )

        return explanations

    def _congestion_from_score(self, score):

        if score < 0.25:
            return "LOW", "GREEN"

        if score < 0.50:
            return "MODERATE", "YELLOW"

        if score < 0.75:
            return "HIGH", "ORANGE"

        return "SEVERE", "RED"

    def _movement_live_congestion(self):
        """
        Dashboard-only visualization signal. Does not affect
        controller decisions - same 70/30 queue/wait blend as
        _queue_live_congestion() below, applied per movement
        instead of per approach.
        """

        congestion = {}

        for movement_id in MOVEMENT_IDS:

            congestion[movement_id] = self._queue_live_congestion(
                self.movement_queues[movement_id]
            )

        return congestion

    def _queue_live_congestion(self, queue):
        """Return the existing display-only congestion score for one queue."""

        average_wait = get_average_wait(
            [vehicle.arrival_tick for vehicle in queue],
            self.tick,
        )

        queue_score = min(len(queue) / 10.0, 1.0)
        wait_score = min(average_wait / 30.0, 1.0)
        score = 0.70 * queue_score + 0.30 * wait_score
        level, colour = self._congestion_from_score(score)

        return {
            "score": float(score),
            "level": level,
            "colour": colour,
        }

    def _lane_live_congestion(self):
        """Display each physical lane without changing movement decisions."""

        return {
            lane_id: self._queue_live_congestion(queue)
            for lane_id, queue in self.lane_queues.items()
        }

    def _approach_live_congestion(self, movement_congestion):
        """
        Mean of the 3 movements' congestion scores for that
        direction - same aggregation choice (mean, not max/sum) as
        phase priority, for consistency and explainability.
        """

        approach_congestion = {}

        for direction in TRAVEL_DIRECTIONS:

            scores = [
                movement_congestion[movement_id]["score"]
                for movement_id in DIRECTION_MOVEMENTS[
                    direction
                ]
            ]

            mean_score = float(np.mean(scores))

            level, colour = self._congestion_from_score(
                mean_score
            )

            approach_congestion[direction] = {
                "score": mean_score,
                "level": level,
                "colour": colour,
            }

        return approach_congestion

    def get_state(self):

        movement_queue_lengths = {
            movement_id: len(
                self.movement_queues[movement_id]
            )
            for movement_id in MOVEMENT_IDS
        }

        lane_queue_lengths = {
            lane_id: len(queue)
            for lane_id, queue in self.lane_queues.items()
        }

        approach_queue_totals = {
            direction: sum(
                movement_queue_lengths[movement_id]
                for movement_id in DIRECTION_MOVEMENTS[
                    direction
                ]
            )
            for direction in TRAVEL_DIRECTIONS
        }

        movement_average_waits = {
            movement_id: float(
                get_average_wait(
                    [
                        v.arrival_tick
                        for v in self.movement_queues[
                            movement_id
                        ]
                    ],
                    self.tick,
                )
            )
            for movement_id in MOVEMENT_IDS
        }

        approach_average_waits = {
            direction: float(
                np.mean(
                    [
                        movement_average_waits[movement_id]
                        for movement_id in DIRECTION_MOVEMENTS[
                            direction
                        ]
                    ]
                )
            )
            for direction in TRAVEL_DIRECTIONS
        }

        movement_signals = self._movement_signals()

        approach_signals = self._approach_signals(
            movement_signals
        )

        movement_live_congestion = (
            self._movement_live_congestion()
        )

        lane_live_congestion = self._lane_live_congestion()

        approach_live_congestion = (
            self._approach_live_congestion(
                movement_live_congestion
            )
        )

        phase_elapsed = (
            self.green_elapsed
            if self.signal_state == "GREEN"
            else self.yellow_elapsed
        )

        # ir_predictions trimmed to JSON-safe, dashboard-relevant
        # fields - top_results (raw retriever hits) are dropped
        # here since they're not JSON-safe and not needed every
        # tick; available via self.ir_predictions directly if a
        # future RAG explanation needs the full evidence.
        ir_predictions_state = {
            movement_id: {
                "severity": float(
                    self.ir_predictions[movement_id][
                        "severity"
                    ]
                ),
                "retrieved_cases": int(
                    self.ir_predictions[movement_id][
                        "retrieved_cases"
                    ]
                ),
                "usable_cases": int(
                    self.ir_predictions[movement_id][
                        "usable_cases"
                    ]
                ),
            }
            for movement_id in MOVEMENT_IDS
        }

        ml_predictions_state = {
            direction: {
                "class": self.ml_predictions[direction][
                    "class"
                ],
                "severity": float(
                    self.ml_predictions[direction][
                        "severity"
                    ]
                ),
            }
            for direction in TRAVEL_DIRECTIONS
        }

        all_completed_waits = [
            wait
            for movement_id in MOVEMENT_IDS
            for wait in self.completed_waits[movement_id]
        ]

        return {
            "tick": self.tick,
            "simulation_duration": SIMULATION_DURATION,
            "finished": self.finished,
            "city": CITY,
            "intersection_id": INTERSECTION_ID,
            "hour": self.hour,
            "weekend": self.weekend,
            "month": self.month,

            "current_phase": self.current_phase,
            "target_phase": self.target_phase,
            "signal_state": self.signal_state,
            "phase_elapsed": phase_elapsed,
            "green_elapsed": self.green_elapsed,
            "yellow_elapsed": self.yellow_elapsed,

            "movement_queues": movement_queue_lengths,
            # Lane state is presentation-only.  ``movement_queues`` above is
            # still the authoritative aggregate supplied to the controller.
            "lane_queues": lane_queue_lengths,
            "approach_queue_totals": approach_queue_totals,
            "movement_average_waits": movement_average_waits,
            "approach_average_waits": approach_average_waits,

            "movement_signals": movement_signals,
            "approach_signals": approach_signals,

            "movement_live_congestion": movement_live_congestion,
            "lane_live_congestion": lane_live_congestion,
            "approach_live_congestion": approach_live_congestion,

            "ml_predictions": ml_predictions_state,
            "ir_predictions": ir_predictions_state,
            "rag_explanations": dict(
                self.rag_explanations
            ),

            "turning_proportions": dict(
                self.turning_proportions
            ),

            "movement_starvation": dict(
                self.movement_starvation
            ),
            "movement_priorities": dict(
                self.movement_priorities
            ),
            "phase_priorities": dict(
                self.phase_priorities
            ),

            "last_decision": dict(self.last_decision),

            "adaptive_switches": self.adaptive_switches,
            "switch_reasons": dict(self.switch_reasons),
            "starvation_override_switches": (
                self.starvation_override_switches
            ),
            "max_starvation_time": self.max_starvation_time,

            "simulated_hour": self.simulated_hour,
            "ml_ir_refresh_history": list(
                self.ml_ir_refresh_history
            ),

            "current_phase_priority": float(
                self.phase_priorities.get(
                    self.current_phase, 0.0
                )
            ),
            "target_phase_priority": (
                float(
                    self.phase_priorities[self.target_phase]
                )
                if self.target_phase is not None
                else None
            ),
            "decision_type": (
                "starvation_override"
                if self.last_decision.get("reason")
                == "starvation_override"
                else "weighted_optimization"
            ),

            "movement_priority_breakdown": {
                movement_id: calculate_movement_priority_breakdown(
                    movement_id=movement_id,
                    movement_queues={
                        m: [
                            v.arrival_tick
                            for v in self.movement_queues[m]
                        ]
                        for m in MOVEMENT_IDS
                    },
                    current_tick=self.tick,
                    ml_predictions=self.ml_predictions,
                    starvation=self.movement_starvation,
                    ir_predictions=self.ir_predictions,
                )
                for movement_id in MOVEMENT_IDS
            },

            "total_arrivals": sum(
                self.movement_total_arrivals.values()
            ),
            "total_departures": sum(
                self.movement_total_departures.values()
            ),
            "movement_total_arrivals": dict(
                self.movement_total_arrivals
            ),
            "movement_total_departures": dict(
                self.movement_total_departures
            ),
            "movement_max_queues": dict(
                self.movement_max_queues
            ),

            "total_waiting": sum(
                movement_queue_lengths.values()
            ),

            # Display-only summary stats for the dashboard - purely
            # a read of the self.completed_waits data already
            # tracked by _serve_green_phase(); no new tracking
            # logic, no controller behavior change.
            "average_completed_wait": (
                float(np.mean(all_completed_waits))
                if all_completed_waits
                else 0.0
            ),
            "max_completed_wait": (
                float(np.max(all_completed_waits))
                if all_completed_waits
                else 0.0
            ),
        }


def print_section(title):
    """Local CLI helper - trivial enough not to warrant adding to
    the shared controller_core.py just for this one script."""

    print(f"\n{title}")


def main():

    print_section(
        "SMART TRAFFIC AI — MOVEMENT-AWARE ADAPTIVE SIMULATION"
    )

    simulator = AdaptiveTrafficSimulation()

    print(
        f"\nRunning {SIMULATION_DURATION} ticks..."
    )

    while not simulator.finished:
        simulator.step()

    state = simulator.get_state()

    print_section("SIMULATION SUMMARY")

    print(f"\nTotal arrivals:   {state['total_arrivals']}")
    print(f"Total departures: {state['total_departures']}")
    print(f"Vehicles waiting: {state['total_waiting']}")
    print(f"Adaptive switches made: {state['adaptive_switches']}")

    print("\nSwitch reasons:")

    if state["switch_reasons"]:

        for reason, count in state["switch_reasons"].items():
            print(f"  {reason}: {count}")

    else:
        print("  No adaptive switches made.")

    print("\nPer-movement statistics:\n")

    for movement_id in MOVEMENT_IDS:

        print(
            f"{movement_id:12s}: "
            f"arrivals={state['movement_total_arrivals'][movement_id]:>4} "
            f"departures={state['movement_total_departures'][movement_id]:>4} "
            f"remaining={state['movement_queues'][movement_id]:>4} "
            f"max_queue={state['movement_max_queues'][movement_id]:>4}"
        )


if __name__ == "__main__":
    main()
