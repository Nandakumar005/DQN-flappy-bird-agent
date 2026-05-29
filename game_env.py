import pygame
import random


WIN_W = 1000
WIN_H = 650
GROUND_H = 80
BIRD_R = 12
PIPE_W = 70
PIPE_GAP = 160
PIPE_SPD = 4
GRAVITY = 0.45
JUMP_VEL = -7.5
PIPE_SPAWN_GAP = 280


class Colors:
    BG = (20, 22, 30)
    GROUND = (40, 42, 50)
    GROUND_LINE = (60, 62, 70)
    PIPE = (0, 180, 60)
    PIPE_DARK = (0, 130, 40)
    PIPE_LIGHT = (0, 220, 80)
    BIRD = (255, 200, 50)
    BIRD_OUTLINE = (200, 150, 30)
    WHITE = (255, 255, 255)
    DARK = (15, 17, 25)
    RED = (255, 80, 80)
    GREEN = (80, 255, 80)
    CYAN = (80, 200, 255)
    GRAY = (150, 150, 150)


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

    def draw(self, screen):
        if not self.alive:
            return
        cx, cy = int(self.x), int(self.y)
        pygame.draw.circle(screen, Colors.BIRD, (cx, cy), BIRD_R)
        pygame.draw.circle(screen, Colors.BIRD_OUTLINE, (cx, cy), BIRD_R, 2)
        eye_x, eye_y = cx + 4, cy - 3
        pygame.draw.circle(screen, Colors.WHITE, (eye_x, eye_y), 4)
        pygame.draw.circle(screen, Colors.DARK, (eye_x + 1, eye_y), 2)
        beak = [
            (cx + BIRD_R, cy),
            (cx + BIRD_R + 8, cy + 2),
            (cx + BIRD_R, cy + 5),
        ]
        pygame.draw.polygon(screen, Colors.RED, beak)


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

    def collides_with(self, bird):
        bl = bird.x - BIRD_R
        br = bird.x + BIRD_R
        bt = bird.y - BIRD_R
        bb = bird.y + BIRD_R
        pl = self.x
        pr = self.x + PIPE_W
        if br > pl and bl < pr:
            if bt < self.gap_top or bb > self.gap_bot:
                return True
        return False

    def draw(self, screen):
        cap_h = 25
        pygame.draw.rect(screen, Colors.PIPE, (self.x, 0, PIPE_W, self.gap_top))
        pygame.draw.rect(
            screen, Colors.PIPE_DARK, (self.x, self.gap_top - cap_h, PIPE_W, cap_h)
        )
        pygame.draw.rect(
            screen, Colors.PIPE_LIGHT, (self.x, self.gap_top - cap_h, PIPE_W, 4)
        )

        bot_top = self.gap_bot
        bot_h = WIN_H - GROUND_H - bot_top
        pygame.draw.rect(screen, Colors.PIPE, (self.x, bot_top, PIPE_W, bot_h))
        pygame.draw.rect(
            screen, Colors.PIPE_DARK, (self.x, bot_top, PIPE_W, cap_h)
        )
        pygame.draw.rect(
            screen, Colors.PIPE_LIGHT, (self.x, bot_top, PIPE_W, 4)
        )


class GameEnv:
    def __init__(self):
        self.birds = []
        self.pipes = []
        self.frame = 0
        self.ground_scroll = 0.0
        self.max_pipe_x = WIN_W
        self.last_gap_center = None

    def add_bird(self, x, y):
        b = Bird(x, y)
        self.birds.append(b)
        return b

    @property
    def alive_birds(self):
        return [b for b in self.birds if b.alive]

    @property
    def alive_count(self):
        return sum(1 for b in self.birds if b.alive)

    @property
    def all_dead(self):
        return self.alive_count == 0

    def get_next_pipe(self, bird):
        for p in self.pipes:
            if not p.passed and p.x + PIPE_W > bird.x:
                return p
        return None

    def update(self):
        self.frame += 1
        self.ground_scroll = (self.ground_scroll - PIPE_SPD) % 24

        if (
            len(self.pipes) == 0
            or self.pipes[-1].x < WIN_W - PIPE_SPAWN_GAP
        ):
            new_pipe = Pipe(WIN_W, self.last_gap_center)
            self.last_gap_center = new_pipe.gap_top + PIPE_GAP // 2
            self.pipes.append(new_pipe)

        for pipe in self.pipes[:]:
            pipe.update()
            if pipe.x + PIPE_W < -20:
                self.pipes.remove(pipe)

        for bird in self.birds:
            if bird.alive:
                bird.update()
                if bird.y + BIRD_R >= WIN_H - GROUND_H:
                    bird.alive = False
                if bird.y - BIRD_R <= 0:
                    bird.alive = False
            else:
                if bird.y + BIRD_R < WIN_H - GROUND_H:
                    bird.vel += GRAVITY
                    bird.y += bird.vel

        for bird in self.birds:
            if not bird.alive:
                continue
            for pipe in self.pipes:
                if pipe.collides_with(bird):
                    bird.alive = False
                    break
                if not pipe.passed and pipe.x + PIPE_W < bird.x:
                    bird.score += 1
                    pipe.passed = True

    def draw_bg(self, screen):
        screen.fill(Colors.BG)
        for i in range(60):
            sx = int((self.frame * 0.2 + i * 137.5) % WIN_W)
            sy = (i * 97 + 53) % (WIN_H - GROUND_H)
            b = 160 + (i % 60)
            screen.set_at((sx, sy), (b, b, b))
        pygame.draw.rect(
            screen,
            Colors.GROUND,
            (0, WIN_H - GROUND_H, WIN_W, GROUND_H),
        )
        pygame.draw.line(
            screen,
            Colors.GROUND_LINE,
            (0, WIN_H - GROUND_H),
            (WIN_W, WIN_H - GROUND_H),
            2,
        )
        for i in range(0, WIN_W + 24, 24):
            x = i - self.ground_scroll
            pygame.draw.rect(
                screen, Colors.GROUND_LINE, (x, WIN_H - GROUND_H + 10, 1, 5)
            )
            pygame.draw.rect(
                screen, Colors.GROUND_LINE, (x + 12, WIN_H - GROUND_H + 25, 1, 5)
            )

    def draw(self, screen):
        self.draw_bg(screen)
        for pipe in self.pipes:
            pipe.draw(screen)
        for bird in self.birds:
            bird.draw(screen)
