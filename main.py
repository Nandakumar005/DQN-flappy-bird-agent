import sys
import os

import neat
import pygame

from game_env import GameEnv, WIN_W, WIN_H, PIPE_GAP, Colors
import visualizer


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")


def make_inputs(bird, game):
    pipe = game.get_next_pipe(bird)
    if pipe is None:
        return [bird.y / WIN_H, bird.vel / 10, 1.0, 0.5]
    return [
        bird.y / WIN_H,
        bird.vel / 10,
        (pipe.x - bird.x) / WIN_W,
        (pipe.gap_top + PIPE_GAP / 2) / WIN_H,
    ]


def main():
    pygame.init()
    pygame.display.set_caption("RL Flappy Bird — NEAT Evolution")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()
    visualizer.init_fonts()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())

    viz = visualizer.Visualizer()
    running = True
    gen_counter = 0

    def eval_genomes(genomes, config):
        nonlocal running, gen_counter
        gen_counter += 1
        gen = gen_counter

        game = GameEnv()
        bird_map = {}

        for genome_id, genome in genomes:
            genome.fitness = 0.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            bird = game.add_bird(120, WIN_H // 2)
            bird_map[bird] = (genome, net)

        timeout = 6000

        while game.alive_count > 0 and game.frame < timeout:
            clock.tick(60 * viz.speed)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        return
                    elif event.key in (pygame.K_UP, pygame.K_EQUALS):
                        viz.speed_up()
                    elif event.key in (pygame.K_DOWN, pygame.K_MINUS):
                        viz.speed_down()

            for bird in game.alive_birds:
                _, net = bird_map[bird]
                output = net.activate(make_inputs(bird, game))
                if output[0] > 0.5:
                    bird.jump()

            game.update()

            for bird, (genome, _) in bird_map.items():
                if bird.alive:
                    genome.fitness = bird.frames + bird.score * 100

            best_genome = max(
                (g for _, g in genomes if g.fitness is not None),
                key=lambda g: g.fitness,
                default=None,
            )

            screen.fill((0, 0, 0))
            game.draw(screen)

            best_fit = int(best_genome.fitness) if best_genome else 0
            viz.draw_hud(screen, gen, game.alive_count, len(genomes), best_fit, viz.speed)
            viz.draw_fitness_graph(screen)
            if best_genome:
                viz.draw_nn(screen, best_genome, config)

            controls = visualizer.FONT_SM.render(
                "[UP]/[DOWN] speed   [R] restart   [ESC] quit",
                True,
                Colors.GRAY,
            )
            screen.blit(controls, (10, WIN_H - 22))

            if game.alive_count == 0 and game.frame <= 10:
                done_txt = pygame.font.SysFont("consolas", 28, bold=True).render(
                    "ALL BIRDS DIED — EVOLVING...", True, (255, 80, 80)
                )
                screen.blit(done_txt, (WIN_W // 2 - 200, WIN_H // 2 - 14))

            pygame.display.flip()

        for bird, (genome, _) in bird_map.items():
            genome.fitness = bird.frames + bird.score * 100

        best_fit = int(max((g.fitness for _, g in genomes), default=0))
        avg_fit = int(
            sum(g.fitness for _, g in genomes) / len(genomes) if genomes else 0
        )
        viz.record_generation(gen, best_fit, avg_fit, game.alive_count, len(genomes))

    while running:
        pop.run(eval_genomes, 1)

        if not running:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    pop = neat.Population(config)
                    pop.add_reporter(neat.StdOutReporter(True))
                    pop.add_reporter(neat.StatisticsReporter())
                    viz = visualizer.Visualizer()
                    gen_counter = 0
                elif event.key == pygame.K_ESCAPE:
                    running = False

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
