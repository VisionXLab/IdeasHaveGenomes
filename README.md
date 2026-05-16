# Ideas Have Genomes: Benchmarking Scientific Lineages through Structured Inheritance

Official code for GENE-BENCH, a benchmark that operationalizes research-idea evolution as genome-centric lineage understanding and generation. GENE-BENCH contains two components:

- **GENE-Exam**: 42 task types and 1,029 closed-form instances evaluating four dimensions of lineage competence — genome abstraction, inheritance mapping, evolutionary reasoning, and lineage validation.
- **GENE-Arena**: 30 domain tasks evaluating lineage-grounded idea generation via Population Evolving Score (PES).

## Quickstart

```bash
pip install -r requirements.txt
```

### Set API Credentials

```bash
export BASE_URL="https://api.openai.com/v1"
export API_KEY="sk-your-key-here"
export MODEL_NAME="gpt-4o"
```

### Run GENE-Exam

```bash
# Smoke test (single task type, 2 instances)
python -m gene_exam.evaluators.eval_benchmark \
  --provider openai \
  --model gpt-4.1-mini \
  --task-type T1-01_contribution_type \
  --max-per-task 2 \
  --output gene_exam/results/smoke.json

# Full 42-task benchmark
python -m gene_exam.evaluators.eval_benchmark \
  --provider openai \
  --model gpt-4.1-mini \
  --concurrency 8 \
  --output gene_exam/results/eval_full.json
```

### Run GENE-Arena PES

Place generated proposals at:

```text
gene_arena/results/<arena-id>/ideas/<task_id>/<participant_id>_<setting>.json
```

Each file should contain:

```json
{
  "content": "... generated proposal text or JSON ..."
}
```

Then score with PES:

```bash
python gene_arena/run_arena.py pes \
  --arena-id smoke \
  --tasks cs_AgentFramework \
  --participants openai-default \
  --settings Question \
  --judge-models judge-gpt4o judge-gpt4o-mini judge-gpt4.1-mini
```

Results are written to `gene_arena/results/<arena-id>/`.

## Repository Structure

```
IdeasHaveGenomes/
├── gene_exam/
│   ├── Questions/           # 42 task types × 1,029 instances
│   └── evaluators/          # Exact-match evaluator
├── gene_arena/
│   ├── task/                # 30 domain tasks (10 domains × 3)
│   ├── run_arena.py         # PES runner
│   ├── dynamics_eval.py     # Evolutionary dynamics inference
│   ├── genome_differ.py     # Gene alignment & diff
│   └── population_evolving_score.py  # PES scoring
├── config.py
└── requirements.txt
```

## License

TBD
