# OnQ project page

Static blog-style page for "Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories".
Open `index.html` directly, or serve the folder (for example with GitHub Pages).

- `static/js/data.js` is generated from the arXiv source tables (`tables/main_results.tex`, `tables/opd_gain.tex`) and the Figure 1 / Dynamic KL data in the paper repo; regenerate it rather than editing by hand if the numbers change.
- All result charts are drawn in `static/js/site.js` from `static/js/data.js`; `static/images/method_overview.svg` is the paper's method diagram.
- Figure 1 and the trace excerpts use the seed-42 Qwen3-1.7B W2 MATH-500 outputs (question "How many zeroes are at the end of 42!?"); token positions were measured with the Qwen3 tokenizer.
- Paper: https://arxiv.org/abs/2610.29378 · Code: https://github.com/Boyuannz/onq (this repo).
