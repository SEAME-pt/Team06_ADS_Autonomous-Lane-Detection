import pygame
from pywidgets.core.layout import Layout
from pywidgets.core.skin import *

class HorizontalBox(Layout):
    def __init__(self, x, y, height, spacing=4):
        super().__init__(x, y, 0, height)
 
        self.spacing = spacing

    def add(self, widget):
        widget.parent = self
        widget.skin = self.skin
        self.widgets.append(widget)
        self._update_layout()
        return widget

    def _update_layout(self):
        x_offset = 0
        for w in self.widgets:
            w.set_position(x_offset, 0)
            w.rect.height = self.rect.height
            x_offset += w.rect.width + self.spacing
        self.rect.width = x_offset - self.spacing if self.widgets else 0

    def update(self, dt):
        super().update(dt)
        self._update_layout()
        for w in self.widgets:
            w.update(dt)

    def render(self, surface):
        for w in self.widgets:
            w.skin = self.skin
            w.render(surface)

    def on_mouse_down(self, x, y):
        for w in reversed(self.widgets):
            if w.on_mouse_down(x, y):
                return True
        return False

    def on_mouse_move(self, x, y):
        for w in reversed(self.widgets):
            if w.on_mouse_move(x, y):
                return True
        return False

    def on_mouse_up(self, x, y):
        for w in self.widgets:
            w.on_mouse_up(x, y)
        return False
    


class HorizontalBoxAuto(Layout):
    def __init__(self, x, y, width, height, spacing=4, margins=None):
        super().__init__(x, y, width, height)
        self.spacing = spacing
        self.entries = []
        self.margins = margins if margins else Margins()

    def set_margins(self, margins):
        self.margins = margins
        self._update_layout()

    def add(self, widget, percent=1.0):
        self.entries.append((widget, percent))
        widget.parent = self
        widget.skin = self.skin
        self.widgets.append(widget)
        self._update_layout()
        return widget

    def _update_layout(self):
        total_spacing = self.spacing * (len(self.entries) - 1)
        usable_width = (self.rect.width-self.rect.x) - self.margins.left - self.margins.right - total_spacing - self.margins.right
        #usable_height =  (self.rect.height-self.rect.y) - self.margins.top - self.margins.bottom - total_spacing- self.margins.top
        x_offset = self.margins.left

        for widget, percent in self.entries:
            widget_width = int(usable_width * percent)
            widget.set_position(x_offset, self.margins.top)
            widget.set_size(widget_width, self.rect.height - self.margins.top - self.margins.bottom)
            x_offset += widget_width + self.spacing

    def on_resize(self, w, h):
        self.rect.width = w 
 
        self._update_layout()
 
