import pygame
import sys
from .manager import WidgetManager

class App:
    def __init__(self, width=800, height=600, title="PyWidgets App", fps=True):
        pygame.init()
        pygame.scrap.init()
        pygame.scrap.set_mode(pygame.SCRAP_CLIPBOARD)

        self.size = (width, height)
        self.screen = pygame.display.set_mode(self.size, pygame.RESIZABLE)
        pygame.display.set_caption(title)

        self.clock = pygame.time.Clock()
        self.show_fps = fps
        self.running = True
        self.manager = WidgetManager(self.screen)
        self.color = (30, 30, 30)
        self.cursor_img = pygame.image.load("cursor.png").convert_alpha()
        self.cursor_img = pygame.transform.smoothscale(self.cursor_img, (22,20))
        pygame.mouse.set_visible(False)
        self.screen.blit(self.cursor_img, pygame.mouse.get_pos())


  


        self.init()
    
    def close(self):
        self.running = False
        self.on_quit()

    def init(self):
        pass

    def handle_event(self, event):
        pass

    def on_quit(self):
        pass

    def on_update(self, dt):
        pass

    def on_render(self, surface):
        pass
    def on_resize(self, w, h):
        pass

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                self._handle_event(event)

            self.on_update(dt)
            self.screen.fill(self.color)


            self.on_render(self.screen)
            self.manager.update(dt)
            self.manager.render()
            #self.manager.debug()

            if self.show_fps:
                self._draw_fps()

            self.screen.blit(self.cursor_img, pygame.mouse.get_pos())
            pygame.display.flip()
        self.close()    
        pygame.quit()
        sys.exit()

    def _handle_event(self, event):
        self.handle_event(event)

        if event.type == pygame.QUIT:
            self.on_quit()
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            self.manager.on_key_down(event.key)
        elif event.type == pygame.KEYUP:
            self.manager.on_key_up(event.key)
        #elif event.type == pygame.TEXTINPUT:
        #    self.manager.on_text_input(event.text)
        elif event.type == pygame.MOUSEMOTION:
            self.manager.on_mouse_move(*event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.manager.on_mouse_down(*event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.manager.on_mouse_up(*event.pos)
        elif event.type == pygame.MOUSEWHEEL:
            self.manager.on_mouse_wheel(event.y)
        elif event.type == pygame.VIDEORESIZE:
            self.on_resize(event.w, event.h)
            self.manager.on_resize(event.w, event.h)

    def _draw_fps(self):
        font = pygame.font.SysFont("Arial", 16)
        fps_text = font.render(f"FPS: {self.clock.get_fps():.1f}", True, (200, 200, 200))
        self.screen.blit(fps_text, (10, 10))
