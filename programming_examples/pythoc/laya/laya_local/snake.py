"""Feature-assisted Snake on the local NPU: python -m laya_local.snake."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

DIRECTIONS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
QUESTIONS = {
    "move": {"type": "choice", "instructions": "Choose the best safe move.", "criteria": list(DIRECTIONS)},
    "safe": {"type": "noul", "instructions": "Is a safe route available?"},
    "reachable": {"type": "noul", "instructions": "Is food reachable through empty cells?"},
    "adjacent": {"type": "noul", "instructions": "Is food adjacent to the head?"},
}


def cycle(width, height):
    """Hamiltonian cycle, with one even board dimension."""
    if width % 2:
        return [(y, x) for x, y in cycle(height, width)]
    cells = [(0, y) for y in range(height)]
    for x in range(1, width):
        cells.extend((x, y) for y in (range(height - 1, 0, -1) if x % 2 else range(1, height)))
    cells.extend((x, 0) for x in range(width - 1, 0, -1))
    return cells


class Game:
    def __init__(self, width=24, height=16, seed=101, initial_length=6):
        for name, value in (("width", width), ("height", height), ("seed", seed), ("initial_length", initial_length)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if not (4 <= width <= 48 and 4 <= height <= 32) or (width % 2 and height % 2):
            raise ValueError("Board must be 4–48 by 4–32, with one even dimension")
        if not 2 <= initial_length < width * height:
            raise ValueError("Initial length must fit the board and be at least 2")
        self.width, self.height, self.seed = width, height, seed
        self.path = cycle(width, height)
        self.index = {point: i for i, point in enumerate(self.path)}
        self.body = list(reversed(self.path[:initial_length]))
        self.rng = random.Random(seed)
        self.steps = self.score = self.interventions = 0
        self.dead = False
        self.food = self._food()

    def _food(self):
        occupied = set(self.body)
        empty = [point for point in self.path if point not in occupied]
        return self.rng.choice(empty) if empty else None

    @property
    def over(self):
        return self.dead or self.food is None

    def target(self, direction):
        dx, dy = DIRECTIONS[direction]
        x, y = self.body[0]
        return x + dx, y + dy

    def legal(self, direction):
        point = self.target(direction)
        x, y = point
        occupied = self.body if point == self.food else self.body[:-1]
        return 0 <= x < self.width and 0 <= y < self.height and point not in occupied

    def safe_moves(self):
        if self.over:
            return []
        head = self.index[self.body[0]]
        size = len(self.path)
        tail_distance = (self.index[self.body[-1]] - head) % size
        food_distance = (self.index[self.food] - head) % size
        return [direction for direction in DIRECTIONS if self.legal(direction)
                and 0 < (self.index[self.target(direction)] - head) % size <= food_distance
                and (self.index[self.target(direction)] - head) % size <= tail_distance]

    def reachable(self):
        blocked = set(self.body[1:])
        seen = {self.body[0]}
        queue = deque(seen)
        while queue:
            x, y = queue.popleft()
            if (x, y) == self.food:
                return True
            for dx, dy in DIRECTIONS.values():
                point = x + dx, y + dy
                if 0 <= point[0] < self.width and 0 <= point[1] < self.height and point not in blocked and point not in seen:
                    seen.add(point)
                    queue.append(point)
        return False

    def prompt(self):
        safe = self.safe_moves()
        head, size = self.index[self.body[0]], len(self.path)
        best = max(safe, key=lambda d: (self.index[self.target(d)] - head) % size) if safe else "none"
        adjacent = any(self.target(d) == self.food for d in DIRECTIONS)
        return (f"Safe moves: {', '.join(safe) or 'none'}. Best: {best}. "
                f"Food reachable: {'yes' if self.reachable() else 'no'}. "
                f"Food adjacent: {'yes' if adjacent else 'no'}.")

    def snapshot(self):
        return dict(width=self.width, height=self.height, seed=self.seed, body=list(self.body),
                    food=self.food, score=self.score, steps=self.steps,
                    interventions=self.interventions, dead=self.dead, won=self.food is None)

    def advance(self, probabilities, shield=True):
        if self.over:
            raise ValueError("Game is over; reset to start again")
        if set(probabilities) != set(DIRECTIONS) or any(
            not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities.values()
        ):
            raise ValueError("Invalid direction probabilities")
        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        safe = self.safe_moves()
        if shield and not safe:
            raise RuntimeError("Cycle invariant broken: no safe successor")
        executed = max(safe, key=probabilities.__getitem__) if shield else proposed
        intervention = executed != proposed
        self.interventions += int(intervention)
        self.steps += 1
        if not self.legal(executed):
            self.dead = True
        else:
            point = self.target(executed)
            self.body.insert(0, point)
            if point == self.food:
                self.score += 1
                self.food = self._food()
            else:
                self.body.pop()
        return dict(proposed=proposed, executed=executed, intervention=intervention, safe=safe)


def step(game, client, shield=True):
    """One fresh synchronous prediction, followed by exactly one game update."""
    if game.over:
        raise ValueError("Game is over; reset to start again")
    before = game.snapshot()
    state = game.prompt()
    started = time.perf_counter()
    result = client.predict(state, QUESTIONS)
    latency = time.perf_counter() - started
    decision = game.advance(result["answers"]["move"]["probabilities"], shield)
    return dict(before=before, game=game.snapshot(), state=state, decision=decision,
                prediction=result, inference_seconds=latency)


def summary(frames, elapsed, game, warmup_seconds, client=None):
    backend = getattr(client, "backend_name", None) or (frames[0].get("prediction", {}).get("routing", {}).get("backend") if frames else None) or "exact"
    samples = sorted(frame["inference_seconds"] for frame in frames)
    return dict(game=game.snapshot(), moves_per_second=len(frames) / elapsed if elapsed else 0,
                elapsed_seconds=elapsed, warmup_seconds=warmup_seconds, samples=len(samples),
                inference_p50_ms=statistics.median(samples) * 1000 if samples else None,
                inference_p95_ms=samples[max(0, math.ceil(.95 * len(samples)) - 1)] * 1000 if samples else None,
                shield=frames[0]["shield"] if frames else None,
                backend=backend, checkpoint="english", question_rows=4,
                timing="Client.predict wall time; includes tokenization, NPU, CPU heads and formatting")


def render(game, frame):
    body = set(game.body)
    rows = []
    for y in range(game.height):
        rows.append(''.join('@@' if (x, y) == game.body[0] else '[]' if (x, y) in body
                            else '()' if (x, y) == game.food else ' .'
                            for x in range(game.width)))
    decision = frame["decision"]
    print('\033[H\033[J' + '\n'.join(rows))
    print(f"Laya {'+ cycle safety' if frame['shield'] else 'raw policy'} | score {game.score} | length {len(game.body)} | moves {game.steps}")
    print(f"{frame['inference_seconds'] * 1000:.1f} ms | {decision['proposed']} → {decision['executed']}"
          f" | shield interventions {game.interventions}")
    print('  '.join(f"{d} {p:.1%}" for d, p in frame['prediction']['answers']['move']['probabilities'].items()))
    print("Ctrl-C to stop", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--initial-length", type=int, default=6)
    parser.add_argument("--fps", type=float, default=12, help="Maximum presentation rate; 0 means uncapped")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--unassisted", action="store_true", help="Disable execution shield; planner features still supplied")
    parser.add_argument("--output", type=Path, help="Save complete inputs, predictions, board states and summary as JSON")
    args = parser.parse_args()
    if args.steps < 1 or not math.isfinite(args.fps) or args.fps < 0:
        parser.error("steps must be positive; fps must be finite and nonnegative")
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new path")
    game = Game(args.width, args.height, args.seed, args.initial_length)
    from . import Client
    frames = []
    with Client() as client:
        print("Warming up the local NPU…", file=sys.stderr, flush=True)
        t = time.perf_counter()
        client.predict(game.prompt(), QUESTIONS)
        warmup_seconds = time.perf_counter() - t
        started = time.perf_counter()
        try:
            while len(frames) < args.steps and not game.over:
                tick = time.perf_counter()
                frame = step(game, client, not args.unassisted)
                frame["shield"] = not args.unassisted
                frames.append(frame)
                if not args.headless:
                    render(game, frame)
                if args.fps and len(frames) < args.steps and not game.over:
                    time.sleep(max(0, 1 / args.fps - (time.perf_counter() - tick)))
        except KeyboardInterrupt:
            pass
        elapsed = time.perf_counter() - started
    report = dict(summary=summary(frames, elapsed, game, warmup_seconds, client),
                  questions=QUESTIONS, target_fps=args.fps, frames=frames)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == "__main__":
    main()
