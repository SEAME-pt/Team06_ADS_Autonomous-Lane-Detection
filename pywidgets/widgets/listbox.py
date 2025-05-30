from pywidgets.core.widget import Widget
from pywidgets.core.skin import *

class ListBox(Widget):
    def __init__(self, x, y, width, height, items=None, on_select=None):
        super().__init__(x, y, width, height)
        self.items = items if items else []
        self.selected_index = -1
        self.on_select = on_select
        self.item_height = 24
        self.scroll_offset = 0
        self.max_visible = self.rect.height // self.item_height
        self.scrollbar_rect = pygame.Rect(0, 0, 8, self.rect.height)
        self.scrollbar_back_rect = pygame.Rect(0, 0, 8, self.rect.height)
        self.dragging = False
        self.over=False
        self.drag_offset = 0

    def update(self, dt):
        self.bound.x = self.get_x()
        self.bound.y = self.get_y()
        self.bound.width = self.rect.width
        self.bound.height = self.rect.height

        self.max_visible = self.rect.height // self.item_height
        self.scrollbar_rect.height = max(16, self.rect.height * self.max_visible // max(1, len(self.items)))
        self.scrollbar_rect.x = self.get_x() + self.rect.width - self.scrollbar_rect.width
        self.scrollbar_back_rect.x = self.get_x() + self.rect.width - self.scrollbar_rect.width
        self.scrollbar_back_rect.y = self.get_y()
        scroll_range = max(1, len(self.items) - self.max_visible)
        self.scrollbar_rect.y = self.get_y() + int((self.scroll_offset / scroll_range) * (self.rect.height - self.scrollbar_rect.height))

    def render(self, surface):
        font = self.skin.font
        text_color = self.skin.get_property(SkinProps.TEXT_COLOR)
        bg_color = self.skin.get_property(SkinProps.BACKGROUND_COLOR)
        border_color = self.skin.get_property(SkinProps.BORDER_COLOR)
        selected_color = self.skin.get_property(SkinProps.BUTTON_HOVER_COLOR)
        scroll_color = self.skin.get_property(SkinProps.SLIDER_HANDLE_COLOR)

        pygame.draw.rect(surface, bg_color, self.bound)
        border =1
        if self.over:
            border=2    
        pygame.draw.rect(surface, border_color, self.bound, border)

        clip = surface.get_clip()
        surface.set_clip(self.bound)

        start = int(self.scroll_offset)
        end = min(start + self.max_visible + 1, len(self.items))
        for i in range(start, end):
            item_y = self.get_y() + (i - start) * self.item_height
            rect = pygame.Rect(self.get_x(), item_y, self.rect.width - 8, self.item_height)
            if i == self.selected_index:
                pygame.draw.rect(surface, selected_color, rect)
            text_surface = font.render(str(self.items[i]), True, text_color)
            surface.blit(text_surface, (rect.x + 4, rect.y + 4))

        surface.set_clip(clip)
        pygame.draw.rect(surface, self.skin.get_property(SkinProps.SCROLLBAR_COLOR), self.scrollbar_back_rect)
        if len(self.items) > self.max_visible:
            pygame.draw.rect(surface, scroll_color, self.scrollbar_rect)

    def on_mouse_down(self, x, y):
        if not self.bound.collidepoint(x, y):
            return False
        self.over = True
        if self.scrollbar_rect.collidepoint(x, y):
            self.dragging = True
            self.drag_offset = y - self.scrollbar_rect.y
            return True

        local_y = y - self.get_y()
        index = int(self.scroll_offset) + local_y // self.item_height
        if 0 <= index < len(self.items):
            self.selected_index = index
            if self.on_select:
                self.on_select(index, self.items[index])
        return True

    def on_mouse_up(self, x, y):
        self.dragging = False
        
        return False

    def on_mouse_move(self, x, y):
        self.over = self.bound.collidepoint(x, y)
        if self.dragging:
            rel_y = y - self.get_y() - self.drag_offset
            scroll_area = self.rect.height - self.scrollbar_rect.height
            rel_y = max(0, min(scroll_area, rel_y))
            percent = rel_y / scroll_area if scroll_area > 0 else 0
            self.scroll_offset = percent * max(0, len(self.items) - self.max_visible)
            return True
        return False

    def on_mouse_wheel(self, delta):
        if  self.over:
            self.scroll_offset = max(0, min(self.scroll_offset - delta, max(0, len(self.items) - self.max_visible)))
            return True
        return False

    def set_items(self, items):
        self.items = items
        self.selected_index = -1
        self.scroll_offset = 0

    def get_selected(self):
        return self.items[self.selected_index] if 0 <= self.selected_index < len(self.items) else None    
    
