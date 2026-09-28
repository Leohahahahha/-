# Findings

- Gemma Attention softmax has [B,K,G,Q,S]; current return is embeddings and cache only. Scan stacks layers. No modality scalar gate exists.
- AFT concatenation is VLM prefix, action (state token for pi0), tactile (history token + H), force (history token + H). Three future streams use identical physical horizon indices.
- Current TCN compresses history to one token; force MLP compresses 8x6 to one token. Attention cannot reveal individual raw history frames/markers.
- AFTPolicy owns fixed-RNG synchronous prediction and physical decoding; correct perturbation boundary is normalized encoded history, not raw marker zeros.
- Serving script already dispatches AFT independently; optional diagnostics can be configured after checkpoint loading, leaving training config/checkpoints unchanged.
- Existing compatibility changes are uncommitted from the prior interrupted task and must be preserved/tested before separate diagnostic commits.

Implementation choice: attention is replayed using the same RNG and Euler schedule at one selected denoising step, independently of standard prediction. This intentionally costs an extra replay pass when enabled but avoids diagnostic buffers/changes in the default training/inference graph. Capture head means at all layers internally; selected layers are exported. Normalized history zeros mean training-statistics mean, not removal of expert or proof of no-touch behavior.
