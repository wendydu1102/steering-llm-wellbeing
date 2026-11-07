# steering-llm-wellbeing
## Repository Organization

The code and materials related to this Q1 checkpoint are in `E-steer_2026/faithful/`. The other three folders contain exploratory work that may be relevant as our Q1 project evolves and will remain in the repository. The `E-steer` reproduction folder has not yet been cleaned up; cleanup is planned.

## AI and Human Contributions

- **AI:** Claude Code, using Claude Opus 5.5 and Sonnet 5.5, wrote nearly all code under `E-steer_2026`, including the baseline proof of concept, the paper-faithful `faithful/` package and notebook, and unit tests. It also drafted the `SCOPE_Q1.md`, and the Q1 checkpoint report and LaTeX files; looked up datasets and citation records; and proposed labeled fallbacks when paper resources were unavailable.
- **Humans:** Chose the paper, research question, and reduced scope; decided on substitutions, including Qwen3-1.7B because the available GPU could not run Qwen3-8B and NRC-VAD labels instead of hand-picked words; ran the GPU experiments; reviewed the results and logs; and responsible for topics, sources, and contents of the report.
- **Testing and status:** Checks included unit tests, a short end-to-end pipeline run, a test of the steering hook on the actual model, and a preliminary alpha-calibration run. The full experiment under the final Q1 scope has not yet been run.
