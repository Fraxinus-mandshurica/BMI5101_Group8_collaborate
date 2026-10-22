"""Section 8: first-hit selector versus corrected MeSH2024 pp=0 baseline."""
import mesh2024_semantic_controls as controls

WALK_SETTINGS = dict(tp_factor=0.3, seed=0, num_walks=100, walk_length=10,
                     p=1, q=1, workers=4, em_max_iter=5)
SEMANTIC_SETTINGS = dict(mesh_year=2024, cutoff=0.4, letter_roots=True)
RUN_SETTINGS = dict(pp=0, tp_factor=0.3, seed=0, dimension=128, window_size=4,
                    epochs=1, workers=4, folds=10, cv_seed=0,
                    walk_length=10, num_walks=100)
RUN_NAME = "msi_seed0_teleport_mesh2024_roots"
REFERENCE_NAME = "msi_seed0_semantic_pp_mesh2024_roots"

assert WALK_SETTINGS == controls.WALK_SETTINGS
assert SEMANTIC_SETTINGS == controls.SEMANTIC_SETTINGS
assert RUN_NAME == controls.TELEPORT_RUN_NAME
assert REFERENCE_NAME == controls.SEMANTIC_RUN_NAME

if __name__ == "__main__":
    controls.main("teleport")
