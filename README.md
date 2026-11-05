# Survey evidence

`front_ends.csv` is Table I of the paper, as data.

Each row records what a published method's **released code** does at
evaluation time, not what its paper says. Two columns carry the provenance:

- `evidence`
  - `C` the evaluation code itself was read
  - `A` no evaluation script is released, so the row reports the attack
    code's front end
  - `P` no code released, so the row comes from the paper
- `commit_read` / `commit_date` the repository head at the time of reading.
  Line numbers quoted in the paper's supplement are that commit's.

`repo_url` is deliberately empty in this first commit. Each URL is filled in
only after it has been checked against the primary source, which is the same
rule the paper's bibliography follows.

Fawkes has a row here but no row in Table I: it ships a protection tool
rather than a benchmark, with no verification threshold and no success-rate
evaluation, so there was nothing to tabulate. LowKey is absent from both
because it distributes a compiled binary and no source.

The four inherited constants, which seven of these repositories carry to six
decimal places:

    IR152 0.166788   IRSE50 0.241045   FaceNet 0.409131   MobileFace 0.301611

Corrections are welcome as pull requests. A correction needs a commit hash
and a file path, not a recollection.
