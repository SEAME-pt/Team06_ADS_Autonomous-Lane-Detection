from pywidgets.core.widget import Widget
from pywidgets.core.skin import *



class Foldout(Widget):
    def __init__(self, x, y, width, title="", expanded=True):
        super().__init__(x, y, width, 24)
        self.title = title
        self.expanded = expanded
        self.children = []
        self.header_height = 24
        self.spacing = 4

    def add(self, widget):
        widget.parent = self
        widget.skin = self.skin
        widget.rect.width = self.rect.width
        self.children.append(widget)
        self._update_layout()
        return widget

    def _update_layout(self):
        y_offset = self.header_height + self.spacing
        for widget in self.children:
            widget.set_position(0, y_offset)
            y_offset += widget.rect.height + self.spacing
        self.rect.height = self.header_height + (y_offset - self.header_height if self.expanded else 0)

    def update(self, dt):
        super().update(dt)
        self._update_layout()
        if self.expanded:
            for widget in self.children:
                widget.update(dt)

    def render(self, surface):
        x, y = self.get_x(), self.get_y()
        font = self.skin.font
        text_color = self.skin.get_property(SkinProps.TEXT_COLOR)
        bg_color = self.skin.get_property(SkinProps.BACKGROUND_COLOR)

        # Header
        pygame.draw.rect(surface, bg_color, (x, y, self.rect.width, self.header_height))
        symbol = "▼" if self.expanded else "▶"
        symbol_surface = font.render(symbol, True, text_color)
        surface.blit(symbol_surface, (x + 4, y + 4))

        title_surface = font.render(self.title, True, text_color)
        surface.blit(title_surface, (x + 20, y + 4))

        # Children
        if self.expanded:
            for widget in self.children:
                widget.skin = self.skin
                widget.render(surface)

    def on_mouse_down(self, x, y):
        if pygame.Rect(self.get_x(), self.get_y(), self.rect.width, self.header_height).collidepoint(x, y):
            self.expanded = not self.expanded
            return True
        if self.expanded:
            for widget in reversed(self.children):
                if widget.on_mouse_down(x, y):
                    return True
        return False

    def on_mouse_up(self, x, y):
        if self.expanded:
            for widget in reversed(self.children):
                widget.on_mouse_up(x, y)
        return False

    def on_mouse_move(self, x, y):
        if self.expanded:
            for widget in reversed(self.children):
                if widget.on_mouse_move(x, y):
                    return True
        return False

    def on_key_down(self, key):
        if self.expanded:
            for widget in self.children:
                widget.on_key_down(key)
        return False

    def on_key_up(self, key):
        if self.expanded:
            for widget in self.children:
                widget.on_key_up(key)
        return False

    def on_resize(self, w, h):
        if self.expanded:
            for widget in self.children:
                widget.on_resize(w, h)    

class FoldoutGroup(Widget):
    def __init__(self, x, y, width):
        super().__init__(x, y, width, 0)
        self.foldouts = []
        self.spacing = 6

    def add_foldout(self, title):
        f = Foldout(0, 0, self.rect.width, title=title, expanded=False)
        f.parent = self
        f.skin = self.skin
        self.foldouts.append(f)
        self._recalculate()
        return f

    def _recalculate(self):
        y_offset = 0
        for f in self.foldouts:
            f.set_position(0, y_offset)
            f.rect.width = self.rect.width
            f._update_layout()
            y_offset += f.rect.height + self.spacing
        self.rect.height = y_offset

    def update(self, dt):
        for f in self.foldouts:
            f.update(dt)
        self._recalculate()

    def render(self, surface):
        for f in self.foldouts:
            f.render(surface)

    def on_mouse_down(self, x, y):
        for f in reversed(self.foldouts):
            if f.on_mouse_down(x, y):
                return True
        return False

    def on_mouse_move(self, x, y):
        for f in reversed(self.foldouts):
            if f.on_mouse_move(x, y):
                return True
        return False

    def on_mouse_up(self, x, y):
        for f in self.foldouts:
            f.on_mouse_up(x, y)
        return False

    def on_key_down(self, key):
        for f in self.foldouts:
            f.on_key_down(key)
        return False
