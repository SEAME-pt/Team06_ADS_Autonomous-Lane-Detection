from pywidgets.core.widget import Widget
from pywidgets.core.skin import *

class Slider(Widget):
    def __init__(self, x, y, length, vertical=False, min_value=0, max_value=100, value=0, on_change=None):
        width, height = (20, length) if vertical else (length, 20)
        super().__init__(x, y, width, height)
        self.vertical = vertical
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.on_change = on_change
        self.dragging = False
        self.over = False
        self.format = ".0f" 
        self.handle_size = 14  
        self.handle_rect = pygame.Rect(0, 0, self.handle_size, self.handle_size)
        self.mode =1

    def update(self, dt):
        super().update(dt)
        self._update_handle_position()

    def _update_handle_position(self):
        pct = (self.value - self.min_value) / (self.max_value - self.min_value)
        if self.vertical:
            y = self.bound.y + pct * (self.bound.height - self.handle_size)
            self.handle_rect.topleft = (self.bound.centerx - self.handle_size // 2, y)
        else:
            x = self.bound.x + pct * (self.bound.width - self.handle_size)
            self.handle_rect.topleft = (x, self.bound.centery - self.handle_size // 2)

    def render(self, surface):
        if not self.visible:
            return

        track_color = self.skin.get_property(SkinProps.SCROLLBAR_COLOR)
        handle_color = self.skin.get_property(SkinProps.SLIDER_HANDLE_COLOR)

        x = self.get_x()
        y = self.get_y()
        w = self.bound.width
        h = self.bound.height
        pygame.draw.rect(surface, track_color, self.bound)
        pygame.draw.rect(surface, handle_color, self.handle_rect)
        if self.over:
            pygame.draw.rect(surface, (0, 0, 0),  self.handle_rect, width=1)
        if self.mode==4:
            return
        text_color = self.skin.get_property(SkinProps.TEXT_COLOR)
        text = f"{self.value:{self.format}}"
        text_surface = self.skin.font.render(text, True, text_color)
        text_width, text_height = self.skin.font.size(text)
        text_x = x 
        text_y = y 
        if  self.vertical:
            if self.mode==0:
                text_x = x +  w 
                text_y = y + (h - text_height) // 2
            elif self.mode==1:
                text_x = x + (w//2) - (text_width//2)
                text_y = y - text_height
            elif self.mode==2:
                text_x = x + (w//2) - (text_width//2)
                text_y = y + h 
        else:
            if self.mode==0:
                text_x = x + (w - text_width) // 2
                text_y = y + (h - text_height) // 2
            elif self.mode==1:
                text_x = x + w 
                text_y = y + (h - text_height) // 2
            elif self.mode==2:
                text_x = x- text_width 
                text_y = y + (h - text_height) // 2
            
        surface.blit(text_surface, (text_x, text_y))



    def on_mouse_down(self, x, y):
        if self.handle_rect.collidepoint(x, y):
            self.dragging = True
            return True
        return False

    def on_mouse_up(self, x, y):
        self.dragging = False
        return False

    def on_mouse_move(self, x, y):
        self.over = self.handle_rect.collidepoint(x, y)
        if not self.dragging:
            return False

        if self.vertical:
            rel_y = y - self.bound.y
            pct = min(max(rel_y / (self.bound.height - self.handle_size), 0), 1)
        else:
            rel_x = x - self.bound.x
            pct = min(max(rel_x / (self.bound.width - self.handle_size), 0), 1)


        new_value = self.min_value + pct * (self.max_value - self.min_value)
        if new_value != self.value:
            self.value = new_value
            if self.on_change:
                self.on_change(self.value)
        return self.over

    def set_value(self, value):
        self.value = max(min(value, self.max_value), self.min_value)
        self._update_handle_position()