from pywidgets.core.widget import Widget
from pywidgets.core.skin import *

class CheckBox(Widget):
    def __init__(self, x, y, size=20, text="", checked=False, on_change=None):
        super().__init__(x, y, size, size)
        self.checked = checked
        self.text = text
        self.on_change = on_change
        self.state = WidgetState.NORMAL

    def render(self, surface):
        if not self.visible:
            return
        x, y = self.get_x(), self.get_y()
        rect = pygame.Rect(x, y, self.rect.width, self.rect.height)


        text_width, text_height = self.skin.font.size(self.text) 
        self.bound.width  = text_width + self.rect.width
        self.bound.height = text_height+ self.rect.height
        self.bound = pygame.Rect(self.get_x(), self.get_y(), self.rect.width, self.rect.height)  

        
        # Caixa
        border_color = self.skin.get_property(SkinProps.BORDER_COLOR)
        fill_color = self.skin.get_property(SkinProps.BACKGROUND_COLOR)
        pygame.draw.rect(surface, fill_color, rect)
        pygame.draw.rect(surface, border_color, rect, width=2)
        # Marca de seleção
        if self.checked:
            check_color = self.skin.get_property(SkinProps.CHECKBOX_CHECK_COLOR, (0, 0, 0))
            pygame.draw.line(surface, check_color, (x+4, y+self.rect.height//2), (x+self.rect.width//2, y+self.rect.height-4), 3)
            pygame.draw.line(surface, check_color, (x+self.rect.width//2, y+self.rect.height-4), (x+self.rect.width-4, y+4), 3)
        # Texto
        if self.text:
            text_surface = self.skin.font.render(self.text, True, self.skin.get_property(SkinProps.TEXT_COLOR))
            surface.blit(text_surface, (x + self.rect.width + 8, y + (self.rect.height - text_surface.get_height()) // 2))

        #pygame.draw.rect(surface, (255,0,255), self.bound, width=1)
        #pygame.draw.rect(surface, (255,0,0), self.rect, width=1)

    def on_mouse_down(self, x, y):
        if self.bound.collidepoint(x, y):
            self.checked = not self.checked
            if self.on_change:
                self.on_change(self.checked)
            return True
        return False

    def is_checked(self):
        return self.checked





class CheckGroup(Widget):
    def __init__(self, x, y, width, height, items,  max_columns=2, spacing=8,  on_change=None):
        super().__init__(x, y, width, height)
        self.items = []
        self.max_columns = max_columns
        self.spacing = spacing
        self.on_change = on_change
        self.selected = 0
        self._create_items(items)
        self._align_items()

    def _create_items(self, labels):
        font = self.skin.font if self.skin else pygame.font.SysFont("Arial", 14)
        self.max_widths = []
        self.max_height = 0

        for idx, label in enumerate(labels):
            text_width, text_height = font.size(label)
            total_width = 24 + 8 + text_width  # 24px para círculo, 8px de espaço
            self.max_widths.append(total_width)
            self.max_height = max(self.max_height, text_height, 24)

            item =  CheckBox(0, 0,20, text=label, on_change=None)
            item.on_change = self.on_change
            item.skin = self.skin
            item.parent = self
            self.items.append(item)
        
 
        self.widget_width = max(self.max_widths)
        self.widget_height = self.max_height + 8  # padding vertical

    def _align_items(self):
 
        for idx, item in enumerate(self.items):
            row = idx // self.max_columns
            col = idx % self.max_columns
            x = col * (self.widget_width + self.spacing)
            y = row * (self.widget_height + self.spacing)
            item.set_position(x, y)

 
        total_cols = min(self.max_columns, len(self.items))
        total_rows = (len(self.items) + self.max_columns - 1) // self.max_columns
        self.set_size(
            total_cols * self.widget_width + (total_cols - 1) * self.spacing,
            total_rows * self.widget_height + (total_rows - 1) * self.spacing
        )

    def select(self, selected_item,value):
        for idx, item in enumerate(self.items):
            
            if idx == selected_item:
                self.selected = idx
                item.checked = value
                if self.on_change:
                    self.on_change(self.selected,item)

 

    def render(self, surface):
        if not self.visible:
            return
        #pygame.draw.rect(surface, (255,0,0), self.bound, width=1)


    
            
        for item in self.items:
            item.skin = self.skin
            item.render(surface)

    def on_mouse_move(self, x, y):
        return super().on_mouse_move(x, y)  

    def on_mouse_down(self, x, y):
        if not self.bound.collidepoint(x, y):
           return False
     
        for idx, item in enumerate(self.items):
            if item.on_mouse_down(x,y):
                self.selected = idx
                if self.on_change:
                    self.on_change(self.selected,item)
                return True


        return False