//! Causal StreamingConv1d — maintains state across chunks, no look-ahead needed.
//! Replaces non-causal Conv1d (centered padding) for streaming inference.
//!
//! # OPT-14 context
//! v1.0's SOLA crossfade introduced algorithmic latency because the Conv1d layers
//! in the ONNX model use *centered* padding (`k//2` left + `k//2` right). The right
//! half is a look-ahead — the model needs future samples before it can emit the
//! current output. SOLA covered this by buffering `sola_search_size` samples and
//! cross-fading chunk boundaries.
//!
//! A *causal* Conv1d uses left-only padding (`k-1` left + `0` right) so every
//! output sample depends only on past + current samples. With no look-ahead,
//! no SOLA crossfade is needed — algorithmic latency drops to zero (one chunk of
//! I/O latency remains, but that is unavoidable for any block-based system).
//!
//! # Algorithm
//!   Regular Conv1d (centered pad): looks at future  → SOLA crossfade required.
//!   Causal Conv1d (left-only pad): looks at past only → no SOLA needed.
//!   For streaming: maintain state buffer of last `(k-1) * channels` samples from
//!   the previous chunk. Each call: prepend state to input → causal Conv1d →
//!   output is the same length as input (no extra latency).
//!
//! # Layout
//! Input/output are `channels * chunk_len` floats. The exact channel layout
//! (interleaved sample-major `[s0c0, s0c1, s1c0, s1c1, ...]` or per-channel
//! `[c0_s0, c0_s1, ..., c1_s0, c1_s1, ...]`) does not matter — state is stored
//! in the same layout as the input, so as long as calls are consistent the
//! state bookkeeping is correct. Tests below use channels=1 for clarity.
//!
//! # Placeholder kernel
//! The actual Conv1d weights are *not* applied here. This module is the state
//! management scaffold. The real implementation will call ONNX `Conv1d` with
//! left-only padding on the `[state | input]` buffer once the streaming ONNX
//! session is wired in (OPT-15). For now `process()` passes input through
//! unchanged so the state buffer bookkeeping is exercised + tested in
//! isolation.

/// Stateful causal 1D convolution for streaming inference.
///
/// Maintains a `(kernel_size - 1) * channels` sample state buffer across calls to
/// [`process`](Self::process). Each call returns a vector the same length as the
/// input — no extra latency is introduced because the convolution is causal
/// (only past + current samples, no look-ahead).
///
/// # Example
/// ```
/// use vc_native::StreamingConv1dState;
///
/// let mut conv = StreamingConv1dState::new(7, 1);
/// let chunk_a = vec![1.0f32; 100];
/// let chunk_b = vec![2.0f32; 100];
/// let out_a = conv.process(&chunk_a);
/// let out_b = conv.process(&chunk_b);
/// assert_eq!(out_a.len(), chunk_a.len());
/// assert_eq!(out_b.len(), chunk_b.len());
/// // state is the last (7-1)=6 samples of chunk_b
/// assert_eq!(conv.state(), &[2.0; 6]);
/// ```
pub struct StreamingConv1dState {
    /// State buffer of last `(kernel_size - 1) * channels` samples, same layout
    /// as the most recent input. Initialized to zeros (left-padding for chunk 0).
    state: Vec<f32>,
    kernel_size: usize,
    channels: usize,
}

impl StreamingConv1dState {
    /// Create a new state buffer for a Conv1d with the given kernel size and
    /// channel count. State is initialized to zeros (the left-padding seen by
    /// the first chunk — equivalent to convolving with a zero-padded past).
    ///
    /// # Panics
    /// Panics if `kernel_size < 1` or `channels < 1`.
    pub fn new(kernel_size: usize, channels: usize) -> Self {
        assert!(kernel_size >= 1, "kernel_size must be >= 1, got {kernel_size}");
        assert!(channels >= 1, "channels must be >= 1, got {channels}");
        Self {
            state: vec![0.0; (kernel_size.saturating_sub(1)) * channels],
            kernel_size,
            channels,
        }
    }

    /// Process a new chunk causally.
    ///
    /// - Input: `input` — `[channels * chunk_len]` floats (interleaved or
    ///   per-channel, just be consistent across calls).
    /// - Output: `Vec<f32>` of length `input.len()` (same length as input).
    ///
    /// The output is causal: `output[i]` depends only on `input[0..=i]` plus
    /// state carried over from previous calls. After this call, state is
    /// updated with the last `(kernel_size - 1) * channels` samples of `input`.
    ///
    /// # Panics
    /// Panics if `input.len()` is not a multiple of `channels`.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let ch = self.channels;
        let state_len = self.kernel_size.saturating_sub(1); // samples per channel
        let state_total = state_len * ch;

        assert!(
            input.len() % ch == 0,
            "input.len() ({}) must be a multiple of channels ({})",
            input.len(),
            ch,
        );

        // Concatenate [state, input] for the convolution. A real ONNX `Conv1d`
        // with left-only padding (k-1 left, 0 right) would see exactly this
        // padded buffer as its unpadded input window.
        let total_len = self.state.len() + input.len();
        let mut buf = vec![0.0f32; total_len];
        buf[..self.state.len()].copy_from_slice(&self.state);
        buf[self.state.len()..].copy_from_slice(input);

        let output_len = input.len();
        let mut output = vec![0.0f32; output_len];

        // ---- Placeholder convolution: pass input through unchanged. ----
        // The real implementation will replace this with an ONNX `Conv1d`
        // call using `buf` as the padded input. For each output position
        // (c, i) the true convolution computes
        //   output[c, i] = bias[c]
        //              + sum_{in_ch=0}^{in_chs-1} sum_{j=0}^{k-1}
        //                  weight[c, in_ch, j] * buf_padded[in_ch, i + j]
        // which is causal in the *unpadded* input because `buf = [state | input]`
        // — every `buf[..., i + j]` for `j in [0, k)` lands in either the state
        // tail or `input[0..=i]`, never in `input[i+1..]`.
        output.copy_from_slice(input);

        // Update state with the last `state_total` samples of `input`.
        if input.len() >= state_total {
            // Common case: input is at least as long as the state buffer —
            // just copy the tail. (memcpy, ~O(k·ch).)
            self.state
                .copy_from_slice(&input[input.len() - state_total..]);
        } else if state_total > 0 {
            // Input shorter than state — shift state left by `input.len()`
            // (dropping the oldest samples) and append the new input at the
            // tail. This preserves the "last state_total samples seen so far"
            // invariant across small chunks.
            let shift = input.len();
            self.state.rotate_left(shift);
            self.state[state_total - shift..].copy_from_slice(input);
        }
        // If state_total == 0 (kernel_size == 1), state is always empty — no-op.

        output
    }

    /// Reset state to zeros. Call this between independent audio streams (e.g.
    /// when switching speakers in a multi-user call) to avoid bleed-through of
    /// the previous stream's tail into the new one.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Read-only access to the current state buffer (mainly for testing /
    /// introspection). Length is `(kernel_size - 1) * channels`.
    pub fn state(&self) -> &[f32] {
        &self.state
    }

    /// Current kernel size.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Current channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_conv1d_state_init() {
        let state = StreamingConv1dState::new(7, 1);
        assert_eq!(state.state.len(), 6); // kernel_size - 1
        assert!(state.state.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_streaming_conv1d_preserves_length() {
        let mut state = StreamingConv1dState::new(7, 1);
        let input = vec![1.0; 100];
        let output = state.process(&input);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_streaming_conv1d_state_updates() {
        let mut state = StreamingConv1dState::new(7, 1);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let _ = state.process(&input);
        // After processing 10 samples with kernel=7, state should be last 6 samples
        assert_eq!(state.state, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn test_streaming_conv1d_reset() {
        let mut state = StreamingConv1dState::new(7, 1);
        state.state[0] = 1.0;
        state.reset();
        assert!(state.state.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_streaming_conv1d_short_input() {
        // Input shorter than state buffer — should handle gracefully
        let mut state = StreamingConv1dState::new(7, 1);
        let input = vec![1.0, 2.0, 3.0]; // only 3 samples, state needs 6
        let output = state.process(&input);
        assert_eq!(output.len(), 3);
    }

    // ---- Extra tests beyond the spec — exercise multichannel + edge cases ----

    #[test]
    fn test_streaming_conv1d_multichannel_state_len() {
        // kernel_size=5, channels=3 → state = (5-1) * 3 = 12 floats
        let state = StreamingConv1dState::new(5, 3);
        assert_eq!(state.state.len(), 12);
        assert_eq!(state.kernel_size(), 5);
        assert_eq!(state.channels(), 3);
    }

    #[test]
    fn test_streaming_conv1d_multichannel_state_update() {
        let mut state = StreamingConv1dState::new(3, 2); // state_len=2, state_total=4
        // 6 samples = 3 frames × 2 ch interleaved
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let _ = state.process(&input);
        // Last state_total (4) samples of input = [3, 4, 5, 6]
        assert_eq!(state.state, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_streaming_conv1d_kernel_size_one() {
        // kernel_size=1 → state is empty (no look-back), output == input
        let mut state = StreamingConv1dState::new(1, 1);
        assert_eq!(state.state.len(), 0);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = state.process(&input);
        assert_eq!(output, input);
        assert_eq!(state.state.len(), 0); // still empty
    }

    #[test]
    fn test_streaming_conv1d_short_input_state_correct() {
        // Two calls of 3 samples each, kernel=7 (state_len=6, state_total=6).
        // After both calls, state should be the last 6 of the combined
        // 6 samples — i.e. exactly the concatenation of the two chunks.
        let mut state = StreamingConv1dState::new(7, 1);
        let _ = state.process(&[1.0, 2.0, 3.0]); // 3 samples, < state_total (6)
        // state was [0;6], shift 3 (still all zeros), append [1,2,3] at [3..6]
        // → [0,0,0,1,2,3]
        assert_eq!(state.state, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);

        let _ = state.process(&[4.0, 5.0, 6.0]); // 3 more samples
        // state was [0,0,0,1,2,3], shift 3 → [1,2,3,0,0,0], append [4,5,6]
        // at [3..6] → [1,2,3,4,5,6]
        assert_eq!(state.state, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_streaming_conv1d_output_is_causal_passthrough() {
        // Placeholder kernel is identity — output should equal input exactly.
        let mut state = StreamingConv1dState::new(7, 1);
        let input: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let output = state.process(&input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_streaming_conv1d_reset_clears_after_process() {
        let mut state = StreamingConv1dState::new(7, 1);
        let _ = state.process(&[1.0; 100]);
        // state is non-zero after processing
        assert!(state.state.iter().any(|&x| x != 0.0));
        state.reset();
        assert!(state.state.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[should_panic(expected = "must be >= 1")]
    fn test_streaming_conv1d_zero_kernel_panics() {
        let _ = StreamingConv1dState::new(0, 1);
    }

    #[test]
    #[should_panic(expected = "must be >= 1")]
    fn test_streaming_conv1d_zero_channels_panics() {
        let _ = StreamingConv1dState::new(7, 0);
    }

    #[test]
    #[should_panic(expected = "must be a multiple of channels")]
    fn test_streaming_conv1d_bad_input_len_panics() {
        let mut state = StreamingConv1dState::new(7, 2); // needs even-length input
        let _ = state.process(&[1.0, 2.0, 3.0]); // 3 is not a multiple of 2
    }
}
