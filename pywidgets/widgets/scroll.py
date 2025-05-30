from pywidgets.core.widget import Widget
from pywidgets.core.skin import *

class Scrollbar(Widget):
    def __init__(self, x, y, length, vertical=True, content_size=100, view_size=50, scroll_pos=0, on_scroll=None):
        super().__init__(x, y, 10 if vertical else length, length if vertical else 10)
        self.vertical = vertical
        self.content_size = content_size
        self.view_size = view_size
        self.scroll_pos = scroll_pos
        self.on_scroll = on_scroll
        self.thumb_drag = False
        self.thumb_offset = 0

    def update(self, dt):
        self.bound = pygame.Rect(self.get_x(), self.get_y(), self.rect.width, self.rect.height)

    def render(self, surface):
        bar_color = self.skin.get_property(SkinProps.SCROLLBAR_COLOR, (100, 100, 100))
        thumb_color = self.skin.get_property(SkinProps.SLIDER_HANDLE_COLOR, (180, 180, 180))

        pygame.draw.rect(surface, bar_color, self.bound)

        size = self.rect.height if self.vertical else self.rect.width
        thumb_size = max(20, self.view_size * size // self.content_size)
        track_length = size - thumb_size
        offset = (self.scroll_pos / max(1, self.content_size - self.view_size)) * track_length

        if self.vertical:
            self.thumb_rect = pygame.Rect(self.get_x(), self.get_y() + offset, self.rect.width, thumb_size)
        else:
            self.thumb_rect = pygame.Rect(self.get_x() + offset, self.get_y(), thumb_size, self.rect.height)

        pygame.draw.rect(surface, thumb_color, self.thumb_rect)

    def on_mouse_down(self, x, y):
        if self.thumb_rect.collidepoint(x, y):
            self.thumb_drag = True
            self.thumb_offset = (y - self.thumb_rect.y) if self.vertical else (x - self.thumb_rect.x)
            return True
        return False

    def on_mouse_up(self, x, y):
        self.thumb_drag = False
        return False

    def on_mouse_move(self, x, y):
        if self.thumb_drag:
            pos = y - self.get_y() - self.thumb_offset if self.vertical else x - self.get_x() - self.thumb_offset
            size = self.rect.height if self.vertical else self.rect.width
            thumb_size = self.thumb_rect.height if self.vertical else self.thumb_rect.width
            track_length = size - thumb_size
            percent = max(0.0, min(1.0, pos / max(1, track_length)))
            self.scroll_pos = percent * (self.content_size - self.view_size)
            if self.on_scroll:
                self.on_scroll(self.scroll_pos)
            return True
        return False

    def on_mouse_wheel(self, delta):
        self.scroll_pos = max(0, min(self.scroll_pos - delta * 20, self.content_size - self.view_size))
        if self.on_scroll:
            self.on_scroll(self.scroll_pos)
        return True

