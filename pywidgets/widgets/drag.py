from pywidgets.core.widget import Widget
from pywidgets.core.skin import *

class DragField(Widget):
    def __init__(self, x, y, width=70, label="", value=0.0, min_value=-1000, max_value=1000, step=1.0, on_change=None):
        super().__init__(x, y, width, 24)
        self.label = label
        self.value = float(value)
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.on_change = on_change
        self.dragging = False
        self.drag_start_x = 0
        self.editing = False
        self.center = False
        self.input_text = str(int(value))
        self.last_click_time = 0

    def render(self, surface):
        font = self.skin.font
        text_color = self.skin.get_property(SkinProps.TEXT_COLOR)
        bg_color = self.skin.get_property(SkinProps.BUTTON_NORMAL_COLOR)
        border_color = self.skin.get_property(SkinProps.BORDER_COLOR)

        x, y = self.get_x(), self.get_y()
        pygame.draw.rect(surface, bg_color, self.bound)
        pygame.draw.rect(surface, border_color, self.bound, 1)

        if self.label:
            label_surface = font.render(self.label, True, text_color)
            text_width, text_height = self.skin.font.size(self.label)
            value_x = x + self.rect.width - text_width - 4
            surface.blit(label_surface, (value_x, y + 4))
        
        if self.editing:
            value_surface = font.render(self.input_text + "|", True, text_color)
        else:
            value_surface = font.render(str(round(self.value, 2)), True, text_color)

        if self.center:
            value_x = x + (self.rect.width - value_surface.get_width()) // 2
            surface.blit(value_surface, (value_x, y + 4))
        else:
            surface.blit(value_surface, (x+4, y + 4))

    def update(self, dt):
        self.bound.x = self.get_x()
        self.bound.y = self.get_y()
        self.bound.width = self.rect.width
        self.bound.height = self.rect.height

    def on_mouse_down(self, x, y):
        if self.bound.collidepoint(x, y):
            now = pygame.time.get_ticks()
            if now - self.last_click_time < 300:
                self.editing = True
                self.input_text = str(round(self.value, 2))
            else:
                self.dragging = True
                self.drag_start_x = x
            self.last_click_time = now
            return True
        return False

    def on_mouse_move(self, x, y):
        if self.dragging:
            dx = x - self.drag_start_x
            if abs(dx) > 1:
                delta = dx * self.step
                self.set_value(self.value + delta)
                self.drag_start_x = x
            return True
        return False

    def on_mouse_up(self, x, y):
        self.dragging = False
        return False

    def on_key_down(self, key):
        if self.editing:
            if key == pygame.K_RETURN:
                try:
                    self.value = float(self.input_text)
                    self.value = max(self.min_value, min(self.max_value, self.value))
                    if self.on_change:
                        self.on_change(self.value)
                except ValueError:
                    pass
                self.editing = False
            elif key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            elif key == pygame.K_MINUS and '-' not in self.input_text:
                self.input_text = '-' + self.input_text
            elif key == pygame.K_PERIOD and '.' not in self.input_text:
                self.input_text += '.'
            else:
                char = pygame.key.name(key)
                if char.isdigit():
                    self.input_text += char
            return True
        return False

    def set_value(self, val):
        new_value = max(self.min_value, min(self.max_value, val))
        if new_value != self.value:
            self.value = new_value
            if self.on_change:
                self.on_change(self.value)

    def get_value(self):
        return self.value
    
