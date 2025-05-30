from pywidgets.core.widget import Widget
from pywidgets.core.skin import *

class ProgressBar(Widget):
    def __init__(self, x, y, width, height, vertical=False, value=0.0, show_text=True, mode=0):
        super().__init__(x, y, width, height)
        self.vertical = vertical
        self.value = max(0.0, min(value, 1.0))  # clamp entre 0 e 1
        self.show_text = show_text
        self.mode = mode  # 0 = centro, 1 = topo, 2 = fora
        self.format = ".0%"  # por padrão mostra como percentagem

    def set_value(self, value):
        self.value = max(0.0, min(value, 1.0))  # clamp
     
    def update(self, dt):
        super().update(dt)

    def render(self, surface):
        if not self.visible:
            return

        # Track (fundo)
        pygame.draw.rect(surface, self.skin.get_property(SkinProps.SCROLLBAR_COLOR), self.bound)

        # Fill (barra de progresso)
        fill_color = self.skin.get_property(SkinProps.PROGRESS_BAR_FILL_COLOR)
        if self.vertical:
            fill_height = int(self.bound.height * self.value)
            fill_rect = pygame.Rect(
                self.bound.x,
                self.bound.y + self.bound.height - fill_height,
                self.bound.width,
                fill_height
            )
        else:
            fill_width = int(self.bound.width * self.value)
            fill_rect = pygame.Rect(
                self.bound.x,
                self.bound.y,
                fill_width,
                self.bound.height
            )

        pygame.draw.rect(surface, fill_color, fill_rect)

        # Texto (se ativado)
        if self.show_text:
            text = f"{self.value:{self.format}}"
            text_surface = self.skin.font.render(text, True, self.skin.get_property(SkinProps.TEXT_COLOR))
            text_width, text_height = self.skin.font.size(text)

            x = self.bound.x + (self.bound.width - text_width) // 2
            y = self.bound.y + (self.bound.height - text_height) // 2

            if self.mode == 1:
                y = self.bound.y - text_height - 2
            elif self.mode == 2:
                y = self.bound.y + self.bound.height + 2

            surface.blit(text_surface, (x, y))