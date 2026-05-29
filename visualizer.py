import pygame

from game_env import Colors, WIN_W, WIN_H, BIRD_R


FONT_SM = None
FONT_MD = None
FONT_LG = None

OVERLAY_ALPHA = 200
PANEL_RADIUS = 12


def init_fonts():
    global FONT_SM, FONT_MD, FONT_LG
    FONT_SM = pygame.font.SysFont("consolas", 14)
    FONT_MD = pygame.font.SysFont("consolas", 18)
    FONT_LG = pygame.font.SysFont("consolas", 22, bold=True)


def draw_rounded_rect(screen, color, rect, radius, alpha=OVERLAY_ALPHA):
    s = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    r = radius
    c = (*color, alpha)
    pygame.draw.rect(s, c, (r, 0, rect.w - 2 * r, rect.h))
    pygame.draw.rect(s, c, (0, r, rect.w, rect.h - 2 * r))
    pygame.draw.circle(s, c, (r, r), r)
    pygame.draw.circle(s, c, (rect.w - r - 1, r), r)
    pygame.draw.circle(s, c, (r, rect.h - r - 1), r)
    pygame.draw.circle(s, c, (rect.w - r - 1, rect.h - r - 1), r)
    screen.blit(s, rect)


class Visualizer:
    def __init__(self):
        self.gen_history = []
        self.alive_history = []
        self.speed = 1
        self.speed_idx = 0
        self.speed_opts = [1, 2, 4, 8, 16]
        self.best_all_time = 0
        self.gen_best = 0
        self.last_alive = 0
        self.last_total = 0

    def speed_up(self):
        self.speed_idx = min(len(self.speed_opts) - 1, self.speed_idx + 1)
        self.speed = self.speed_opts[self.speed_idx]

    def speed_down(self):
        self.speed_idx = max(0, self.speed_idx - 1)
        self.speed = self.speed_opts[self.speed_idx]

    def record_generation(self, gen, best, avg, alive, total):
        self.gen_history.append((gen, best, avg))
        self.alive_history.append(alive)
        if best > self.best_all_time:
            self.best_all_time = best
        self.gen_best = best
        self.last_alive = alive
        self.last_total = total
        if len(self.gen_history) > 100:
            self.gen_history.pop(0)
            self.alive_history.pop(0)

    def draw_hud(self, screen, gen, alive, total, best_gen, speed):
        rect = pygame.Rect(10, 10, 220, 120)
        draw_rounded_rect(screen, (10, 10, 18), rect, PANEL_RADIUS)

        lines = [
            f"GEN: {gen}",
            f"ALIVE: {alive}/{total}",
            f"BEST: {best_gen}",
            f"ALL-TIME: {self.best_all_time}",
            f"SPEED: {speed}x",
        ]
        for i, line in enumerate(lines):
            color = Colors.CYAN if i == 0 else Colors.WHITE
            if i == 3:
                color = Colors.GREEN
            if i == 4:
                color = Colors.GRAY
            txt = FONT_MD.render(line, True, color)
            screen.blit(txt, (rect.x + 12, rect.y + 8 + i * 20))

    def draw_fitness_graph(self, screen):
        rect = pygame.Rect(10, WIN_H - 130, 730, 120)
        draw_rounded_rect(screen, (10, 10, 18), rect, PANEL_RADIUS, 180)

        data = self.gen_history
        if len(data) < 2:
            label = FONT_SM.render("Fitness history (waiting for data...)", True, Colors.GRAY)
            screen.blit(label, (rect.x + 10, rect.y + 8))
            return

        plot_x = rect.x + 50
        plot_y = rect.y + 15
        plot_w = rect.w - 60
        plot_h = rect.h - 35

        max_fit = max(b for _, b, _ in data) or 1
        n = len(data)

        pygame.draw.rect(
            screen, (30, 32, 42), (plot_x, plot_y, plot_w, plot_h)
        )
        for i in range(5):
            y = plot_y + plot_h * i // 4
            pygame.draw.line(
                screen, (50, 52, 60), (plot_x, y), (plot_x + plot_w, y), 1
            )
            val = max_fit * (4 - i) // 4
            lbl = FONT_SM.render(str(val), True, Colors.GRAY)
            screen.blit(lbl, (plot_x - 40, y - 7))

        pts_best = []
        pts_avg = []
        for j, (_, best, avg) in enumerate(data):
            x = plot_x + (j / (n - 1)) * plot_w
            yb = plot_y + plot_h - (best / max_fit) * plot_h
            ya = plot_y + plot_h - (avg / max_fit) * plot_h
            pts_best.append((int(x), int(yb)))
            pts_avg.append((int(x), int(ya)))

        if len(pts_best) > 1:
            pygame.draw.lines(screen, Colors.CYAN, False, pts_best, 2)
            pygame.draw.lines(screen, Colors.GREEN, False, pts_avg, 2)

        label_best = FONT_SM.render("best", True, Colors.CYAN)
        label_avg = FONT_SM.render("avg", True, Colors.GREEN)
        screen.blit(label_best, (plot_x + plot_w - 60, plot_y + 2))
        screen.blit(label_avg, (plot_x + plot_w - 30, plot_y + 2))

        gen_label = FONT_SM.render("generation", True, Colors.GRAY)
        screen.blit(gen_label, (plot_x + plot_w // 2 - 30, plot_y + plot_h + 2))

    def draw_nn(self, screen, genome, config):
        rect = pygame.Rect(WIN_W - 250, 10, 240, 300)
        draw_rounded_rect(screen, (10, 10, 18), rect, PANEL_RADIUS, 200)

        if genome is None or config is None:
            txt = FONT_MD.render("No genome data", True, Colors.GRAY)
            screen.blit(txt, (rect.x + 30, rect.y + 130))
            return

        ik = set(config.genome_config.input_keys)
        ok = set(config.genome_config.output_keys)
        inputs = [nid for nid in genome.nodes if nid in ik]
        outputs = [nid for nid in genome.nodes if nid in ok]
        hidden = [nid for nid in genome.nodes if nid not in ik and nid not in ok]

        label = FONT_SM.render(
            f"Net: {len(inputs)}i/{len(hidden)}h/{len(outputs)}o  conns: {len(genome.connections)}",
            True,
            Colors.WHITE,
        )
        screen.blit(label, (rect.x + 8, rect.y + 6))

        px = rect.x + 20
        py = rect.y + 30
        pw = rect.w - 40
        ph = rect.h - 40

        pos = {}
        in_cnt = len(inputs)
        out_cnt = len(outputs)
        for i, nid in enumerate(inputs):
            x = px + pw * (i + 1) / (in_cnt + 1)
            y = py + ph - 15
            pos[nid] = (x, y)
        for i, nid in enumerate(outputs):
            x = px + pw * (i + 1) / (out_cnt + 1)
            y = py + 15
            pos[nid] = (x, y)
        for i, nid in enumerate(hidden):
            x = px + pw / 2
            y = py + ph * (i + 1) / (len(hidden) + 1)
            pos[nid] = (x, y)

        conn_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        for conn in genome.connections.values():
            if conn.enabled and conn.key[0] in pos and conn.key[1] in pos:
                w = conn.weight
                thick = max(1, min(5, int(abs(w) * 0.8 + 1)))
                r = int(255 * (1 if w > 0 else 0))
                g = int(255 * (1 if w < 0 else 0))
                b = 50
                a = min(255, int(150 + abs(w) * 10))
                p1 = (int(pos[conn.key[0]][0]), int(pos[conn.key[0]][1]))
                p2 = (int(pos[conn.key[1]][0]), int(pos[conn.key[1]][1]))
                pygame.draw.line(conn_surf, (r, g, b, a), p1, p2, thick)
        screen.blit(conn_surf, (0, 0))

        for nid, (x, y) in pos.items():
            if nid in ik:
                color = (100, 180, 255)
                r = 6
            elif nid in ok:
                color = (255, 180, 100)
                r = 7
            else:
                color = (200, 200, 200)
                r = 5
            pygame.draw.circle(screen, color, (int(x), int(y)), r)
            pygame.draw.circle(screen, Colors.WHITE, (int(x), int(y)), r, 1)

        in_lbl = FONT_SM.render("I", True, (100, 180, 255))
        out_lbl = FONT_SM.render("O", True, (255, 180, 100))
        screen.blit(in_lbl, (rect.x + 6, py + ph - 14))
        screen.blit(out_lbl, (rect.x + 6, py + 18))
