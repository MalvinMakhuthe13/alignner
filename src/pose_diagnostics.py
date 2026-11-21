
import numpy as np


def pose_constraint_diagnostics(pairs, poses, components):
    """Measure relative-pose constraint residuals without modifying poses."""
    rotation_errors = []
    translation_errors = []
    component_errors = {}

    component_lookup = {}
    for component_index, component in enumerate(components or []):
        for camera_id in component:
            component_lookup[camera_id] = component_index

    skipped = 0

    for pair in pairs:
        if pair.R is None or pair.t is None:
            skipped += 1
            continue

        if pair.idx1 not in poses or pair.idx2 not in poses:
            skipped += 1
            continue

        try:
            Ri, ti = poses[pair.idx1]
            Rj, tj = poses[pair.idx2]

            R_measured = np.asarray(pair.R, dtype=np.float64).reshape(3, 3)
            t_measured = np.asarray(pair.t, dtype=np.float64).reshape(3)

            R_predicted = np.asarray(Rj) @ np.asarray(Ri).T

            relative_rotation = R_measured.T @ R_predicted
            cosine = np.clip(
                (np.trace(relative_rotation) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
            rotation_deg = float(np.degrees(np.arccos(cosine)))

            predicted_translation = (
                np.asarray(tj).reshape(3)
                - R_predicted @ np.asarray(ti).reshape(3)
            )

            measured_norm = np.linalg.norm(t_measured)
            predicted_norm = np.linalg.norm(predicted_translation)

            if not np.isfinite(rotation_deg):
                skipped += 1
                continue

            rotation_errors.append(rotation_deg)

            translation_deg = None
            if measured_norm > 1e-10 and predicted_norm > 1e-10:
                cosine = np.clip(
                    np.dot(predicted_translation, t_measured)
                    / (predicted_norm * measured_norm),
                    -1.0,
                    1.0,
                )
                translation_deg = float(np.degrees(np.arccos(cosine)))
                translation_errors.append(translation_deg)

            component_index = component_lookup.get(pair.idx1, -1)

            bucket = component_errors.setdefault(
                component_index,
                {"rotation": [], "translation": []},
            )
            bucket["rotation"].append(rotation_deg)

            if translation_deg is not None:
                bucket["translation"].append(translation_deg)

        except (ValueError, TypeError, np.linalg.LinAlgError):
            skipped += 1

    def summarize(values):
        if not values:
            return {
                "count": 0,
                "median_deg": None,
                "p90_deg": None,
                "p95_deg": None,
                "max_deg": None,
            }

        values = np.asarray(values, dtype=np.float64)

        return {
            "count": int(len(values)),
            "median_deg": float(np.median(values)),
            "p90_deg": float(np.percentile(values, 90)),
            "p95_deg": float(np.percentile(values, 95)),
            "max_deg": float(np.max(values)),
        }

    return {
        "edges_evaluated": len(rotation_errors),
        "edges_skipped": skipped,
        "rotation": summarize(rotation_errors),
        "translation_direction": summarize(translation_errors),
        "components": {
            str(index): {
                "rotation": summarize(values["rotation"]),
                "translation_direction": summarize(values["translation"]),
            }
            for index, values in sorted(component_errors.items())
        },
    }
