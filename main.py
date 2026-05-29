import os
import sys
import argparse
import pygame
import numpy as np

from game_env import FlappyBirdEnv, load_sprites, draw_sprite_frame, draw_score, WIN_W, WIN_H, GROUND_H, GRAVITY
from dqn_agent import DQNAgent


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
N_EPISODES = 1000
REPORT_EVERY = 10


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    env = FlappyBirdEnv()
    agent = DQNAgent()

    best_reward = -float("inf")
    recent_rewards = []

    header = f"{'Ep':>6} | {'Reward':>8} | {'Eps':>6} | {'Avg':>8} | {'Best':>8} | {'Score':>5}"
    print(header)
    print("-" * len(header))

    for ep in range(1, N_EPISODES + 1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.train()

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        avg_reward = np.mean(recent_rewards) if recent_rewards else total_reward

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(MODEL_PATH)

        if ep % REPORT_EVERY == 0 or ep == 1:
            print(f"{ep:6d} | {total_reward:8.0f} | {agent.epsilon:6.4f} | {avg_reward:8.1f} | {best_reward:8.0f} | {env.bird.score:5d}", flush=True)

    print(f"\nTraining complete! Best reward: {best_reward:.0f}", flush=True)
    print(f"Model saved to: {MODEL_PATH}\n", flush=True)


def play():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Flappy Bird - DQN Agent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 18, bold=True)

    sprites = load_sprites()
    if sprites is None:
        print("ERROR: Could not load sprites. Check sprites_repo/sprites/ directory.")
        pygame.quit()
        sys.exit(1)

    env = FlappyBirdEnv(max_frames=None)
    agent = DQNAgent()
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0

    state = env.reset()
    running = True
    game_over = False
    death_timer = 0

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    state = env.reset()
                    game_over = False
                    death_timer = 0

        if game_over:
            env.bird.vel += GRAVITY
            env.bird.y += env.bird.vel
            death_timer += 1
            if death_timer > 45:
                state = env.reset()
                game_over = False
                death_timer = 0
        else:
            action = agent.act(state, eval_mode=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                game_over = True

        draw_sprite_frame(screen, env, sprites)
        draw_score(screen, env.bird.score, sprites)

        info_lines = [
            f"Score: {env.bird.score}",
            f"Frames: {env.frame}",
        ]
        for i, line in enumerate(info_lines):
            txt = font.render(line, True, (255, 255, 255))
            screen.blit(txt, (12, 12 + i * 22))

        controls = font.render("R: reset | ESC: quit", True, (255, 255, 255))
        screen.blit(controls, (12, WIN_H - GROUND_H - 30))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


def main():
    parser = argparse.ArgumentParser(description="Flappy Bird - DQN")
    parser.add_argument("--play", action="store_true", help="Skip training, load best model and play")
    args = parser.parse_args()

    if args.play:
        play()
        return

    print("=" * 50, flush=True)
    print("  Flappy Bird - DQN Training", flush=True)
    print("=" * 50, flush=True)
    train()
    print("=" * 50, flush=True)
    print("  Starting visual playback", flush=True)
    print("=" * 50, flush=True)
    play()


if __name__ == "__main__":
    main()
