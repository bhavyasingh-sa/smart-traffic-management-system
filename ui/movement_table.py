"""ui/movement_table.py - the all-12-movements detail table. No expander of its own: Streamlit can't nest expanders, and app.py already wraps this in one."""

from simulation.movement_definitions import MOVEMENT_IDS, PHASES

from ui.styles import render_html


def render_movement_table(state):

    active_movements = set(
        PHASES[state["current_phase"]]
    )

    rows = []

    for movement_id in MOVEMENT_IDS:

        is_active = movement_id in active_movements

        row_class = (
            "stc-row-active" if is_active else ""
        )

        rows.append(
            f"""
            <tr class="{row_class}">
                <td>{movement_id}</td>
                <td>
                    {state['movement_queues'][movement_id]}
                </td>
                <td>
                    {state['movement_average_waits'][movement_id]:.1f}s
                </td>
                <td>
                    {state['movement_priorities'][movement_id]:.4f}
                </td>
                <td>
                    {state['movement_starvation'][movement_id]}
                </td>
            </tr>
            """
        )

    render_html(f"""
    <div class="stc-panel">
        <div class="stc-panel-title">
            All 12 Movements
            (highlighted = currently active phase:
            {state['current_phase']})
        </div>
        <table class="stc-table">
            <thead>
                <tr>
                    <th>Movement</th>
                    <th>Queue</th>
                    <th>Average Wait</th>
                    <th>Priority</th>
                    <th>Starvation</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """)
