import pygame
import random
import os

WIN_W = 1000
WIN_H = 650
GROUND_H = 80
PIPE_W = 70
PIPE_GAP = 160
PIPE_SPD = 4
GRAVITY = 0.45
JUMP_VEL = -7.5
PIPE_SPAWN_GAP = 280
BIRD_R = 12

SPRITES_DIR = os.path.join(os.path.dirname(__file__), "sprites_repo", "sprites")


class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vel = 0.0
        self.alive = True
        self.score = 0
        self.frames = 0

    def jump(self):
        if self.alive:
            self.vel = JUMP_VEL

    def update(self):
        if not self.alive:
            return
        self.vel += GRAVITY
        self.y += self.vel
        self.frames += 1


class Pipe:
    def __init__(self, x, prev_gap_center=None):
        self.x = x
        if prev_gap_center is None:
            gap_center = random.randint(220, 420)
        else:
            lo = max(PIPE_GAP // 2 + 50, prev_gap_center - 80)
            hi = min(WIN_H - GROUND_H - PIPE_GAP // 2 - 50, prev_gap_center + 80)
            gap_center = random.randint(lo, hi)
        self.gap_top = gap_center - PIPE_GAP // 2
        self.gap_bot = gap_center + PIPE_GAP // 2
        self.passed = False

    def update(self):
        self.x -= PIPE_SPD

    def collides_with(self, bx, by, br):
        bl, brr = bx - br, bx + br
        bt, bb = by - br, by + br
        pl, pr = self.x, self.x + PIPE_W
        if brr > pl and bl < pr:
            if bt < self.gap_top or bb > self.gap_bot:
                return True
        return False

    @property
    def gap_center(self):
        return (self.gap_top + self.gap_bot) // 2


class FlappyBirdEnv:
    def __init__(self, max_frames=6000):
        self.max_frames = max_frames
        self.bird = None
        self.pipes = []
        self.frame = 0
        self.last_gap_center = None

    def reset(self):
        self.bird = Bird(120, WIN_H // 2)
        self.pipes = []
        self.frame = 0
        self.last_gap_center = None
        first_pipe = Pipe(220, None)
        self.last_gap_center = first_pipe.gap_center
        self.pipes.append(first_pipe)
        return self._state()

    def _next_pipe(self):
        for p in self.pipes:
            if not p.passed and p.x + PIPE_W > self.bird.x:
                return p
        return None

    def _state(self):
        pipe = self._next_pipe()
        if pipe is None:
            return [0.5, 0.0, 1.0, 0.5]
        return [
            self.bird.y / WIN_H,
            self.bird.vel / 10,
            (pipe.x - self.bird.x) / WIN_W,
            pipe.gap_center / WIN_H,
        ]

    def step(self, action):
        if action == 1 and self.bird.alive:
            self.bird.jump()

        self.frame += 1
        self.bird.update()

        if len(self.pipes) == 0 or self.pipes[-1].x < WIN_W - PIPE_SPAWN_GAP:
            np_ = Pipe(WIN_W, self.last_gap_center)
            self.last_gap_center = np_.gap_center
            self.pipes.append(np_)

        for pipe in self.pipes[:]:
            pipe.update()
            if pipe.x + PIPE_W < -20:
                self.pipes.remove(pipe)

        done = False
        reward = 0.1

        for pipe in self.pipes:
            if not pipe.passed and pipe.x + PIPE_W < self.bird.x:
                self.bird.score += 1
                pipe.passed = True
                reward += 10

        if self.bird.y + BIRD_R >= WIN_H - GROUND_H or self.bird.y - BIRD_R <= 0:
            self.bird.alive = False
            done = True
        elif self.max_frames is not None and self.frame >= self.max_frames:
            done = True
        else:
            for pipe in self.pipes:
                if pipe.collides_with(self.bird.x, self.bird.y, BIRD_R):
                    self.bird.alive = False
                    done = True
                    break

        if done:
            if self.max_frames is not None and self.frame >= self.max_frames:
                reward = self.bird.score * 10 + 10
            else:
                reward = -1

        return self._state(), reward, done, {"score": self.bird.score}


def load_sprites():
    try:
        bg = pygame.image.load(os.path.join(SPRITES_DIR, "background-day.png"))
        bg = pygame.transform.scale(bg, (WIN_W, WIN_H - GROUND_H))
        ground = pygame.image.load(os.path.join(SPRITES_DIR, "base.png"))
        ground = pygame.transform.scale(ground, (WIN_W + 48, GROUND_H))
        pipe_img = pygame.image.load(os.path.join(SPRITES_DIR, "pipe-green.png"))
        pipe_head_h = min(30, pipe_img.get_height() // 3)
        pipe_head = pipe_img.subsurface((0, 0, pipe_img.get_width(), pipe_head_h))
        pipe_body = pipe_img.subsurface((0, pipe_head_h, pipe_img.get_width(), pipe_img.get_height() - pipe_head_h))
        pipe_head = pygame.transform.scale(pipe_head, (PIPE_W, int(pipe_head_h * PIPE_W / pipe_img.get_width())))
        pipe_head_h_scaled = pipe_head.get_height()
        bird_imgs = [
            pygame.image.load(os.path.join(SPRITES_DIR, "yellowbird-upflap.png")),
            pygame.image.load(os.path.join(SPRITES_DIR, "yellowbird-midflap.png")),
            pygame.image.load(os.path.join(SPRITES_DIR, "yellowbird-downflap.png")),
        ]
        bird_scale = 1.8
        bird_sz = (int(bird_imgs[0].get_width() * bird_scale), int(bird_imgs[0].get_height() * bird_scale))
        bird_imgs = [pygame.transform.scale(b, bird_sz) for b in bird_imgs]
        nums = {}
        for i in range(10):
            img = pygame.image.load(os.path.join(SPRITES_DIR, f"{i}.png"))
            nums[i] = img
        return {
            "bg": bg,
            "ground": ground,
            "pipe_head": pipe_head,
            "pipe_head_h": pipe_head_h_scaled,
            "pipe_body": pipe_body,
            "bird_imgs": bird_imgs,
            "nums": nums,
        }
    except Exception as e:
        print(f"Sprite loading failed ({e}), using colored rendering")
        return None


def draw_sprite_frame(screen, env, sprites):
    s = sprites
    screen.blit(s["bg"], (0, 0))

    ground_scroll = (env.frame * PIPE_SPD) % (WIN_W + 48)
    screen.blit(s["ground"], (-ground_scroll, WIN_H - GROUND_H))
    screen.blit(s["ground"], (-ground_scroll + WIN_W + 48, WIN_H - GROUND_H))

    pipe_body = s["pipe_body"]
    pipe_head = s["pipe_head"]
    pipe_head_h = s["pipe_head_h"]

    for pipe in env.pipes:
        top_h = pipe.gap_top
        body_h = top_h - pipe_head_h
        if body_h > 0:
            body_scaled = pygame.transform.scale(pipe_body, (PIPE_W, body_h))
            body_scaled = pygame.transform.flip(body_scaled, False, True)
            screen.blit(body_scaled, (pipe.x, 0))
        head_flipped = pygame.transform.flip(pipe_head, False, True)
        screen.blit(head_flipped, (pipe.x, top_h - pipe_head_h))

        bot_top = pipe.gap_bot
        bot_h = WIN_H - GROUND_H - bot_top
        body_h_bot = bot_h - pipe_head_h
        screen.blit(pipe_head, (pipe.x, bot_top))
        if body_h_bot > 0:
            body_scaled = pygame.transform.scale(pipe_body, (PIPE_W, body_h_bot))
            screen.blit(body_scaled, (pipe.x, bot_top + pipe_head_h))

    if env.bird:
        bird_idx = 1
        if env.bird.vel < -2:
            bird_idx = 0
        elif env.bird.vel > 2:
            bird_idx = 2
        bird_img = s["bird_imgs"][bird_idx]
        if not env.bird.alive:
            bird_img = s["bird_imgs"][2]
        bw, bh = bird_img.get_size()
        screen.blit(bird_img, (env.bird.x - bw // 2, env.bird.y - bh // 2))


def draw_score(screen, score, sprites):
    digits = [int(d) for d in str(score)]
    total_w = sum(sprites["nums"][d].get_width() for d in digits)
    x = (WIN_W - total_w) // 2
    for d in digits:
        screen.blit(sprites["nums"][d], (x, 50))
        x += sprites["nums"][d].get_width()
