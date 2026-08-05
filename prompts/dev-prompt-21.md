# Document deferred execution architecture

## What to implement

- Create a root `TODO.md` that records the architecture agreed upon for safely
  batching dynamic eager-mode operations in one command buffer.
- Cover the separation between reusable operators, per-call dispatch plans,
  multi-dispatch execution plans, the executor, and in-flight command batches.

## How to implement

- Describe ownership and responsibilities for shader pipelines, buffer
  bindings, push constants, dispatch dimensions, descriptor pools and sets,
  barriers, submissions, and resource lifetimes.
- Record the descriptor-set issue in the current design and the intended
  per-dispatch allocation and fence-based recycling model.
- Include eager-mode enqueue/flush behavior, automatic buffer hazard tracking,
  and multi-pass operations such as Softmax.
- Organize the work as actionable implementation phases and acceptance checks.
- Do not modify source files or run builds/tests.
