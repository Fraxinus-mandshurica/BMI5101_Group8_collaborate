"""Section 7: original semantic walks with corrected MeSH2024 roots, pp=1/0."""
import mesh2024_semantic_controls as controls

WALK_SETTINGS = dict(tp_factor=0.3, seed=0, num_walks=100, walk_length=10,
                     p=1, q=1, workers=4, em_max_iter=5)
SEMANTIC_SETTINGS = dict(mesh_year=2024, cutoff=0.4, letter_roots=True)
RUN_NAME = "msi_seed0_semantic_pp_mesh2024_roots"

assert WALK_SETTINGS == controls.WALK_SETTINGS
assert SEMANTIC_SETTINGS == controls.SEMANTIC_SETTINGS
assert RUN_NAME == controls.SEMANTIC_RUN_NAME

if __name__ == "__main__":
    controls.main("semantic")
