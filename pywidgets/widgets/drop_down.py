from pywidgets.core.widget import Widget
from pywidgets.core.skin import *

class DropDown(Widget):
    def __init__(self, x, y, width, options, selected=0, on_change=None):
        super().__init__(x, y, width, 24)
        self.options = options
        self.selected = selected
        self.on_change = on_change
        self.expanded = False
        self.over=False
        self.hovered_index = -1

    def update(self, dt):
        super().update(dt)
        self.bound.height = 24 + len(self.options) * 24 if self.expanded else 24

    def render(self, surface):
        if not self.visible:
            return

        x, y = self.get_x(), self.get_y()
        font = self.skin.font
        text_color = self.skin.get_property(SkinProps.TEXT_COLOR)
        bg_color = self.skin.get_property(SkinProps.DROPDOWN_BACKGROUND, self.skin.get_property(SkinProps.BACKGROUND_COLOR))
        hover_color = self.skin.get_property(SkinProps.DROPDOWN_HOVER, self.skin.get_property(SkinProps.BUTTON_HOVER_COLOR))
        border_color = self.skin.get_property(SkinProps.DROPDOWN_BORDER, self.skin.get_property(SkinProps.BORDER_COLOR))

        pygame.draw.rect(surface, bg_color, (x, y, self.rect.width, 24))
        pygame.draw.rect(surface, border_color, (x, y, self.rect.width, 24), 1)

        selected_text = font.render(self.options[self.selected], True, text_color)
        surface.blit(selected_text, (x + 6, y + 4))

        if self.expanded:
            pygame.draw.polygon(surface, text_color, [
                (x + self.rect.width - 6, y + 10),
                (x + self.rect.width - 12, y + 10),
                (x + self.rect.width - 9, y + 14)
            ])
        else:
            pygame.draw.polygon(surface, text_color, [
                (x + self.rect.width - 12, y + 10),
                (x + self.rect.width - 6, y + 10),
                (x + self.rect.width - 9, y + 14)
            ])

        if self.expanded:
            for idx, opt in enumerate(self.options):
                oy = y + 24 + idx * 24
                rect = pygame.Rect(x, oy, self.rect.width, 24)
                color = hover_color if idx == self.hovered_index else bg_color
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, border_color, rect, 1)
                text = font.render(opt, True, text_color)
                surface.blit(text, (x + 6, oy + 4))

    def on_mouse_down(self, x, y):
        if not self.bound.collidepoint(x, y):
            self.expanded = False
            self.over=False
            return False

        rel_y = y - self.get_y()
        if rel_y < 24:
            self.expanded = not self.expanded
        elif self.expanded:
            idx = (rel_y - 24) // 24
            if 0 <= idx < len(self.options):
                self.selected = idx
                self.expanded = False
                if self.on_change:
                    self.on_change(self.options[self.selected])
        return True

    def on_mouse_move(self, x, y):
        if not self.expanded:
            self.hovered_index = -1
            return False
        if self.bound.collidepoint(x, y):
            self.over=True
            rel_y = y - self.get_y()
            if 24 <= rel_y < 24 + len(self.options) * 24:
                self.hovered_index = (rel_y - 24) // 24
            else:
                self.hovered_index = -1
            return True
        else:
            self.over=False
            self.hovered_index = -1
            self.expanded = False
        return False
    
    def on_mouse_up(self, x, y):
        self.over=False
        return False  

    def get_selected(self):
        return self.options[self.selected]

    def set_selected(self, index):
        if 0 <= index < len(self.options):
            self.selected = index