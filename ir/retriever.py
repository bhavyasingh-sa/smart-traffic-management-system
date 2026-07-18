import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from ir.text_processor import clean_text


CONGESTION_SEVERITY_MAP = {
    "low": 0.10,
    "moderate": 0.30,
    "high": 0.55,
    "severe": 0.85,
}


class TrafficRetriever:

    def __init__(
        self,
        documents,
        metadata,
        inverted_index,
    ):
        """
        Hybrid historical traffic information retriever.

        Retrieval pipeline:

        1. Clean historical documents.
        2. Build TF-IDF document matrix.
        3. Use inverted index for candidate generation.
        4. Apply mandatory city/intersection filtering.
        5. Optionally apply strict traffic-context filtering.
        6. Rank candidates using TF-IDF cosine similarity.
        7. Add metadata-aware context bonuses.
        8. Return top-ranked historical traffic cases.
        """

        self.documents = documents

        self.metadata = metadata

        self.inverted_index = inverted_index

        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
        )

        cleaned_documents = [
            clean_text(document)
            for document in documents
        ]

        print(
            "\nBuilding TF-IDF matrix..."
        )

        self.document_matrix = (
            self.vectorizer.fit_transform(
                cleaned_documents
            )
        )

        print(
            "TF-IDF matrix created successfully."
        )

        print(
            f"Matrix shape: "
            f"{self.document_matrix.shape}"
        )


    @staticmethod
    def _normalize_text(value):
        """
        Safely normalize a metadata value into lowercase text.
        """

        if value is None:
            return None

        return str(value).strip().lower()


    @staticmethod
    def _normalize_approach(value):
        """
        Normalize traffic approach values.

        Examples:

        W     -> W
        west  -> W
        NW    -> NW
        north -> N
        """

        if value is None:
            return None

        value = str(
            value
        ).strip().upper()

        heading_map = {
            "NORTH": "N",
            "NORTHEAST": "NE",
            "EAST": "E",
            "SOUTHEAST": "SE",
            "SOUTH": "S",
            "SOUTHWEST": "SW",
            "WEST": "W",
            "NORTHWEST": "NW",
        }

        return heading_map.get(
            value,
            value,
        )


    @staticmethod
    def _safe_int(value):
        """
        Safely convert a value to integer.
        """

        if value is None:
            return None

        try:
            return int(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):
            return None


    @staticmethod
    def _safe_float(value):
        """
        Safely convert a value to float.
        """

        if value is None:
            return None

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None


    def _get_metadata_approach(
        self,
        metadata,
    ):
        """
        Get the approach direction from metadata.

        Prefer the explicit Approach field.

        Fall back to EntryHeading for compatibility with
        older IR models.
        """

        approach = metadata.get(
            "Approach"
        )

        if approach is None:
            approach = metadata.get(
                "EntryHeading"
            )

        return self._normalize_approach(
            approach
        )


    def _apply_legacy_filters(
        self,
        candidate_ids,
        filters,
    ):
        """
        Support older calls that pass a filters dictionary.

        Example:

        filters={
            "City": "Atlanta",
            "IntersectionId": 84,
        }

        This keeps the retriever backward-compatible.
        """

        if not filters:
            return candidate_ids

        filtered_ids = []

        for doc_id in candidate_ids:

            metadata = self.metadata[
                int(doc_id)
            ]

            matches = True

            for key, expected_value in (
                filters.items()
            ):

                actual_value = metadata.get(
                    key
                )

                if (
                    self._normalize_text(
                        actual_value
                    )
                    !=
                    self._normalize_text(
                        expected_value
                    )
                ):
                    matches = False
                    break

            if matches:
                filtered_ids.append(
                    int(doc_id)
                )

        return np.array(
            filtered_ids,
            dtype=int,
        )


    def _apply_location_filter(
        self,
        candidate_ids,
        city=None,
        intersection_id=None,
    ):
        """
        Apply city and intersection filtering.

        These are treated as strong location constraints.
        """

        if (
            city is None
            and intersection_id is None
        ):
            return candidate_ids

        normalized_city = (
            self._normalize_text(city)
        )

        normalized_intersection = (
            self._safe_int(
                intersection_id
            )
        )

        filtered_ids = []

        for doc_id in candidate_ids:

            metadata = self.metadata[
                int(doc_id)
            ]

            if normalized_city is not None:

                metadata_city = (
                    self._normalize_text(
                        metadata.get(
                            "City"
                        )
                    )
                )

                if (
                    metadata_city
                    != normalized_city
                ):
                    continue

            if (
                normalized_intersection
                is not None
            ):

                metadata_intersection = (
                    self._safe_int(
                        metadata.get(
                            "IntersectionId"
                        )
                    )
                )

                if (
                    metadata_intersection
                    != normalized_intersection
                ):
                    continue

            filtered_ids.append(
                int(doc_id)
            )

        return np.array(
            filtered_ids,
            dtype=int,
        )


    def _metadata_matches(
        self,
        metadata,
        approach=None,
        hour=None,
        weekend=None,
        month=None,
        congestion_level=None,
    ):
        """
        Check whether one historical traffic case matches
        the requested structured traffic context.

        Used when strict_context=True.
        """

        if approach is not None:

            expected_approach = (
                self._normalize_approach(
                    approach
                )
            )

            actual_approach = (
                self._get_metadata_approach(
                    metadata
                )
            )

            if (
                actual_approach
                != expected_approach
            ):
                return False

        if hour is not None:

            expected_hour = (
                self._safe_int(hour)
            )

            actual_hour = (
                self._safe_int(
                    metadata.get(
                        "Hour"
                    )
                )
            )

            if (
                actual_hour
                != expected_hour
            ):
                return False

        if weekend is not None:

            expected_weekend = (
                self._safe_int(
                    weekend
                )
            )

            actual_weekend = (
                self._safe_int(
                    metadata.get(
                        "Weekend"
                    )
                )
            )

            if (
                actual_weekend
                != expected_weekend
            ):
                return False

        if month is not None:

            expected_month = (
                self._safe_int(month)
            )

            actual_month = (
                self._safe_int(
                    metadata.get(
                        "Month"
                    )
                )
            )

            if (
                actual_month
                != expected_month
            ):
                return False

        if congestion_level is not None:

            expected_congestion = (
                self._normalize_text(
                    congestion_level
                )
            )

            actual_congestion = (
                self._normalize_text(
                    metadata.get(
                        "CongestionLevel"
                    )
                )
            )

            if (
                actual_congestion
                != expected_congestion
            ):
                return False

        return True


    def _apply_strict_context_filter(
        self,
        candidate_ids,
        approach=None,
        hour=None,
        weekend=None,
        month=None,
        congestion_level=None,
    ):
        """
        Keep only candidates that exactly match all supplied
        structured traffic context values.
        """

        filtered_ids = []

        for doc_id in candidate_ids:

            metadata = self.metadata[
                int(doc_id)
            ]

            matches = (
                self._metadata_matches(
                    metadata=metadata,
                    approach=approach,
                    hour=hour,
                    weekend=weekend,
                    month=month,
                    congestion_level=(
                        congestion_level
                    ),
                )
            )

            if matches:
                filtered_ids.append(
                    int(doc_id)
                )

        return np.array(
            filtered_ids,
            dtype=int,
        )


    def _calculate_context_bonus(
        self,
        metadata,
        approach=None,
        hour=None,
        weekend=None,
        month=None,
        congestion_level=None,
        target_severity=None,
    ):
        """
        Calculate metadata-aware reranking bonus.

        Maximum exact-match bonuses:

        Approach:          +0.12
        Hour:              +0.08
        Weekend:           +0.06
        Month:             +0.04
        Congestion level:  +0.12

        Historical severity can additionally contribute
        up to +0.08 based on closeness to target severity.
        """

        bonus = 0.0

        if approach is not None:

            expected_approach = (
                self._normalize_approach(
                    approach
                )
            )

            actual_approach = (
                self._get_metadata_approach(
                    metadata
                )
            )

            if (
                actual_approach
                == expected_approach
            ):
                bonus += 0.12

        if hour is not None:

            expected_hour = (
                self._safe_int(hour)
            )

            actual_hour = (
                self._safe_int(
                    metadata.get(
                        "Hour"
                    )
                )
            )

            if (
                actual_hour
                == expected_hour
            ):
                bonus += 0.08

            elif (
                actual_hour is not None
                and expected_hour is not None
            ):

                hour_difference = min(
                    abs(
                        actual_hour
                        - expected_hour
                    ),
                    24 - abs(
                        actual_hour
                        - expected_hour
                    ),
                )

                if hour_difference == 1:
                    bonus += 0.04

                elif hour_difference == 2:
                    bonus += 0.02

        if weekend is not None:

            expected_weekend = (
                self._safe_int(
                    weekend
                )
            )

            actual_weekend = (
                self._safe_int(
                    metadata.get(
                        "Weekend"
                    )
                )
            )

            if (
                actual_weekend
                == expected_weekend
            ):
                bonus += 0.06

        if month is not None:

            expected_month = (
                self._safe_int(month)
            )

            actual_month = (
                self._safe_int(
                    metadata.get(
                        "Month"
                    )
                )
            )

            if (
                actual_month
                == expected_month
            ):
                bonus += 0.04

        if congestion_level is not None:

            expected_congestion = (
                self._normalize_text(
                    congestion_level
                )
            )

            actual_congestion = (
                self._normalize_text(
                    metadata.get(
                        "CongestionLevel"
                    )
                )
            )

            if (
                actual_congestion
                == expected_congestion
            ):
                bonus += 0.12

        if target_severity is not None:

            target_value = (
                self._safe_float(
                    target_severity
                )
            )

            historical_value = (
                self._safe_float(
                    metadata.get(
                        "HistoricalSeverity"
                    )
                )
            )

            if (
                target_value is not None
                and historical_value is not None
            ):

                target_value = min(
                    max(
                        target_value,
                        0.0,
                    ),
                    1.0,
                )

                historical_value = min(
                    max(
                        historical_value,
                        0.0,
                    ),
                    1.0,
                )

                difference = abs(
                    target_value
                    - historical_value
                )

                severity_similarity = max(
                    0.0,
                    1.0 - difference,
                )

                bonus += (
                    severity_similarity
                    * 0.08
                )

        return float(bonus)


    def search(
        self,
        query,
        top_k=5,
        filters=None,
        city=None,
        intersection_id=None,
        approach=None,
        hour=None,
        weekend=None,
        month=None,
        congestion_level=None,
        target_severity=None,
        strict_context=False,
        candidate_limit=None,
    ):
        """
        Retrieve and rank historical traffic cases.

        Parameters
        ----------
        query:
            Natural-language traffic query.

        top_k:
            Number of final results to return.

        filters:
            Optional legacy metadata filters dictionary.

        city:
            Optional city constraint.

        intersection_id:
            Optional intersection constraint.

        approach:
            Optional traffic approach:
            N, S, E, W, NE, NW, SE, SW.

        hour:
            Optional hour from 0 to 23.

        weekend:
            Optional day type:
            0 = weekday
            1 = weekend.

        month:
            Optional month from 1 to 12.

        congestion_level:
            Optional congestion category:
            Low, Moderate, High, Severe.

        target_severity:
            Optional continuous severity target from 0 to 1.

        strict_context:
            If True, exact structured context matching is
            applied before ranking.

        candidate_limit:
            Optional limit on the number of highest TF-IDF
            candidates that receive metadata reranking.
        """

        top_k = max(
            int(top_k),
            1,
        )

        candidates = (
            self.inverted_index
            .get_candidates(
                query
            )
        )

        if not candidates:
            return []

        candidate_ids = np.array(
            sorted(candidates),
            dtype=int,
        )

        candidate_ids = (
            self._apply_legacy_filters(
                candidate_ids=(
                    candidate_ids
                ),
                filters=filters,
            )
        )

        if len(candidate_ids) == 0:
            return []

        candidate_ids = (
            self._apply_location_filter(
                candidate_ids=(
                    candidate_ids
                ),
                city=city,
                intersection_id=(
                    intersection_id
                ),
            )
        )

        if len(candidate_ids) == 0:
            return []

        if strict_context:

            candidate_ids = (
                self._apply_strict_context_filter(
                    candidate_ids=(
                        candidate_ids
                    ),
                    approach=approach,
                    hour=hour,
                    weekend=weekend,
                    month=month,
                    congestion_level=(
                        congestion_level
                    ),
                )
            )

            if len(candidate_ids) == 0:
                return []

        cleaned_query = clean_text(
            query
        )

        query_vector = (
            self.vectorizer.transform(
                [cleaned_query]
            )
        )

        candidate_matrix = (
            self.document_matrix[
                candidate_ids
            ]
        )

        similarities = (
            candidate_matrix
            @ query_vector.T
        ).toarray().ravel()

        ranked_positions = np.argsort(
            similarities
        )[::-1]

        if candidate_limit is not None:

            candidate_limit = max(
                int(candidate_limit),
                top_k,
            )

            ranked_positions = (
                ranked_positions[
                    :candidate_limit
                ]
            )

        ranked_results = []

        for position in ranked_positions:

            tfidf_score = float(
                similarities[
                    position
                ]
            )

            if tfidf_score <= 0:
                continue

            doc_id = int(
                candidate_ids[
                    position
                ]
            )

            metadata = self.metadata[
                doc_id
            ]

            context_bonus = (
                self._calculate_context_bonus(
                    metadata=metadata,
                    approach=approach,
                    hour=hour,
                    weekend=weekend,
                    month=month,
                    congestion_level=(
                        congestion_level
                    ),
                    target_severity=(
                        target_severity
                    ),
                )
            )

            final_score = (
                tfidf_score
                + context_bonus
            )

            result = {
                "doc_id": doc_id,
                "score": float(
                    final_score
                ),
                "tfidf_score": float(
                    tfidf_score
                ),
                "context_bonus": float(
                    context_bonus
                ),
                "document": self.documents[
                    doc_id
                ],
                "metadata": metadata,
            }

            ranked_results.append(
                result
            )

        ranked_results.sort(
            key=lambda result: (
                result["score"]
            ),
            reverse=True,
        )

        return ranked_results[
            :top_k
        ]