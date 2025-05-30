
from pywidgets.core.widget import Widget
from pywidgets.core.skin import *
import pygame

class TextBox(Widget):
    def __init__(self, x, y, width=150, height=24, text="", on_change=None, placeholder=""):
        super().__init__(x, y, width, height)
        self.text = text
        self.on_change = on_change
        self.placeholder = placeholder
        self.active = False
        self.cursor_pos = len(text)
        self.cursor_visible = True
        self.cursor_timer = 0.0
        self.cursor_blink_time = 0.5
        self.max_length = 1024
        self.scroll_x = 0
        
        # Seleção de texto
        self.selection_start = -1
        self.selection_end = -1
        self.mouse_selecting = False
        self.double_click_time = 0.3
        self.last_click_time = 0
        
        # Histórico para undo/redo
        self.history = [text]
        self.history_index = 0
        self.max_history = 50
        
        # Validação e formatação
        self.validator = None  # Função para validar entrada
        self.formatter = None  # Função para formatar texto
        self.input_type = "text"  # "text", "number", "email", "password"
        self.password_char = "*"

    def _add_to_history(self):

        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        self.history.append(self.text)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.history_index += 1

    def _undo(self):
 
        if self.history_index > 0:
            self.history_index -= 1
            self.text = self.history[self.history_index]
            self.cursor_pos = min(self.cursor_pos, len(self.text))

    def _redo(self):

        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.text = self.history[self.history_index]
            self.cursor_pos = min(self.cursor_pos, len(self.text))

    def _delete_selection(self):

        if self.selection_start != -1 and self.selection_end != -1:
            start = min(self.selection_start, self.selection_end)
            end = max(self.selection_start, self.selection_end)
            self.text = self.text[:start] + self.text[end:]
            self.cursor_pos = start
            self.selection_start = self.selection_end = -1
            return True
        return False

    def _get_selected_text(self):

        if self.selection_start != -1 and self.selection_end != -1:
            start = min(self.selection_start, self.selection_end)
            end = max(self.selection_start, self.selection_end)
            return self.text[start:end]
        return ""

    def _get_cursor_pos_from_mouse(self, mouse_x):
    
        font = self.skin.font
        relative_x = mouse_x - self.get_x() - 4 + self.scroll_x
        
        if relative_x <= 0:
            return 0
        
        for i in range(len(self.text) + 1):
            char_x = font.size(self.text[:i])[0]
            if char_x >= relative_x:
                return i
        
        return len(self.text)

    def _select_word_at_cursor(self):
 
        if not self.text:
            return
            
        # Encontrar início da palavra
        start = self.cursor_pos
        while start > 0 and self.text[start-1].isalnum():
            start -= 1
            
        # Encontrar fim da palavra  
        end = self.cursor_pos
        while end < len(self.text) and self.text[end].isalnum():
            end += 1
            
        if start < end:
            self.selection_start = start
            self.selection_end = end

    def _validate_input(self, text):

        if self.validator:
            return self.validator(text)
            
        if self.input_type == "number":
            try:
                float(text) if text else 0
                return True
            except ValueError:
                return False
        elif self.input_type == "email":
            return "@" in text or not text
        
        return True

    def _format_text(self, text):
        if self.formatter:
            return self.formatter(text)
        return text

    def _get_display_text(self):
        if self.input_type == "password" and self.text:
            return self.password_char * len(self.text)
        return self.text

    def _get_char_from_key(self, key, shift_pressed):
        """Converte código de tecla para caractere"""
        
        # Letras (a-z)
        if pygame.K_a <= key <= pygame.K_z:
            char = chr(key)
            return char.upper() if shift_pressed else char
        
        # Números no teclado principal (0-9)
        if pygame.K_0 <= key <= pygame.K_9:
            if shift_pressed:
                shift_numbers = {
                    pygame.K_0: ')', pygame.K_1: '!', pygame.K_2: '@', pygame.K_3: '#', 
                    pygame.K_4: '$', pygame.K_5: '%', pygame.K_6: '^', pygame.K_7: '&', 
                    pygame.K_8: '*', pygame.K_9: '('
                }
                return shift_numbers.get(key, str(key - pygame.K_0))
            else:
                return str(key - pygame.K_0)
        
        # Números do teclado numérico
        if pygame.K_KP0 <= key <= pygame.K_KP9:
            return str(key - pygame.K_KP0)
        
        # Símbolos e pontuação
        symbol_map = {
            pygame.K_PERIOD: '.' if not shift_pressed else '>',
            pygame.K_COMMA: ',' if not shift_pressed else '<',
            pygame.K_SEMICOLON: ';' if not shift_pressed else ':',
            pygame.K_QUOTE: "'" if not shift_pressed else '"',
            pygame.K_SLASH: '/' if not shift_pressed else '?',
            pygame.K_BACKSLASH: '\\' if not shift_pressed else '|',
            pygame.K_LEFTBRACKET: '[' if not shift_pressed else '{',
            pygame.K_RIGHTBRACKET: ']' if not shift_pressed else '}',
            pygame.K_MINUS: '-' if not shift_pressed else '_',
            pygame.K_EQUALS: '=' if not shift_pressed else '+',
            pygame.K_BACKQUOTE: '`' if not shift_pressed else '~',
            pygame.K_KP_PERIOD: '.',
            pygame.K_KP_DIVIDE: '/',
            pygame.K_KP_MULTIPLY: '*',
            pygame.K_KP_MINUS: '-',
            pygame.K_KP_PLUS: '+',
        }
        
        if key in symbol_map:
            return symbol_map[key]
        
        # Teclas especiais que não devem inserir caracteres
        special_keys = {
            pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_TAB, pygame.K_BACKSPACE, 
            pygame.K_DELETE, pygame.K_INSERT, pygame.K_HOME, pygame.K_END, 
            pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, 
            pygame.K_LSHIFT, pygame.K_RSHIFT, pygame.K_LCTRL, pygame.K_RCTRL, 
            pygame.K_LALT, pygame.K_RALT, pygame.K_ESCAPE
        }
        
        if key in special_keys:
            return None
        
        return None

    def update(self, dt):
        super().update(dt)
        self.bound.x = self.get_x()
        self.bound.y = self.get_y()
        self.bound.width = self.rect.width
        self.bound.height = self.rect.height
        
        # Cursor piscando
        self.cursor_timer += dt
        if self.cursor_timer > self.cursor_blink_time:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0.0

        # Auto-scroll horizontal
        font = self.skin.font
        cursor_pixel = font.size(self.text[:self.cursor_pos])[0]
        text_area_width = self.rect.width - 16  # Margem
        
        if cursor_pixel - self.scroll_x > text_area_width:
            self.scroll_x = cursor_pixel - text_area_width
        elif cursor_pixel < self.scroll_x:
            self.scroll_x = cursor_pixel

    def render(self, surface):
        font = self.skin.font
        text_color = self.skin.get_property(SkinProps.TEXT_COLOR)
        placeholder_color = (150, 150, 150)
        bg_color = self.skin.get_property(SkinProps.BACKGROUND_COLOR)
        border_color = self.skin.get_property(SkinProps.BORDER_COLOR)
        selection_color = (100, 150, 255, 100)  # Azul semi-transparente
        focus_border_color = (100, 150, 255)  # Azul para foco

        x, y = self.get_x(), self.get_y()
        
        # Fundo
        pygame.draw.rect(surface, bg_color, self.bound)
        
        # Borda (diferente se ativo)
        border_col = focus_border_color if self.active else border_color
        pygame.draw.rect(surface, border_col, self.bound, 2 if self.active else 1)

        # Configurar clipping
        clip = surface.get_clip()
        text_rect = pygame.Rect(x + 4, y, self.rect.width - 8, self.rect.height)
        surface.set_clip(text_rect)

        # Determinar texto e cor para renderização
        display_text = self._get_display_text()
        if display_text or self.active:
            text_to_render = display_text
            color = text_color
        else:
            text_to_render = self.placeholder
            color = placeholder_color

        # Renderizar seleção
        if (self.selection_start != -1 and self.selection_end != -1 and self.active):
            start = min(self.selection_start, self.selection_end)
            end = max(self.selection_start, self.selection_end)
            
            if start < end:
                sel_start_x = font.size(display_text[:start])[0]
                sel_end_x = font.size(display_text[:end])[0]
                
                sel_rect = pygame.Rect(
                    x + 4 + sel_start_x - self.scroll_x,
                    y + 4,
                    sel_end_x - sel_start_x,
                    font.get_height()
                )
                
                # Criar surface temporária para transparência
                temp_surface = pygame.Surface((sel_rect.width, sel_rect.height), pygame.SRCALPHA)
                temp_surface.fill(selection_color)
                surface.blit(temp_surface, (sel_rect.x, sel_rect.y))

        # Renderizar texto
        if text_to_render:
            text_surface = font.render(text_to_render, True, color)
            text_x = x + 4 - self.scroll_x
            surface.blit(text_surface, (text_x, y + 4))

        # Renderizar cursor
        if self.active and self.cursor_visible and display_text:
            cursor_x = x + 4 + font.size(display_text[:self.cursor_pos])[0] - self.scroll_x
            cursor_y = y + 4
            pygame.draw.line(surface, text_color, 
                           (cursor_x, cursor_y), (cursor_x, cursor_y + font.get_height()), 2)
        elif self.active and self.cursor_visible and not display_text:
            # Cursor no início quando não há texto
            cursor_x = x + 4
            cursor_y = y + 4
            pygame.draw.line(surface, text_color, 
                           (cursor_x, cursor_y), (cursor_x, cursor_y + font.get_height()), 2)

        surface.set_clip(clip)

    def on_mouse_down(self, x, y):
        current_time = pygame.time.get_ticks() / 1000.0
        
        if self.bound.collidepoint(x, y):
            self.active = True
            
            # Duplo clique para selecionar tudo
            if current_time - self.last_click_time < self.double_click_time:
                self.selection_start = 0
                self.selection_end = len(self.text)
            else:
                self.cursor_pos = self._get_cursor_pos_from_mouse(x)
                self.selection_start = self.selection_end = -1
                self.mouse_selecting = True
            
            self.last_click_time = current_time
            return True
        else:
            if self.active and self.on_change:
                self.on_change(self.text)
            self.active = False
            return False

    def on_mouse_up(self, x, y):
        self.mouse_selecting = False
        return False

    def on_mouse_move(self, x, y):
        if self.mouse_selecting and self.active:
            new_pos = self._get_cursor_pos_from_mouse(x)
            
            if self.selection_start == -1:
                self.selection_start = self.cursor_pos
            self.selection_end = new_pos
            self.cursor_pos = new_pos
            return True
        return False

    def on_key_down(self, key):
        if not self.active:
            return False

        keys = pygame.key.get_pressed()
        ctrl_pressed = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]
        shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        # Atalhos com Ctrl
        if ctrl_pressed:
            if key == pygame.K_z:
                self._undo()
                return True
            elif key == pygame.K_y:
                self._redo()
                return True
            elif key == pygame.K_a:
                self.selection_start = 0
                self.selection_end = len(self.text)
                return True
            elif key == pygame.K_c:
                selected = self._get_selected_text()
                if selected:
                    try:
                        pygame.scrap.put(pygame.SCRAP_TEXT, selected.encode())
                    except pygame.error:
                        pass
                return True
            elif key == pygame.K_x:
                selected = self._get_selected_text()
                if selected:
                    try:
                        pygame.scrap.put(pygame.SCRAP_TEXT, selected.encode())
                    except pygame.error:
                        pass
                    self._add_to_history()
                    self._delete_selection()
                return True
            elif key == pygame.K_v:
                try:
                    clip = pygame.scrap.get(pygame.SCRAP_TEXT)
                    if clip:
                        pasted = clip.decode(errors="ignore").replace('\x00', '')
                        if self._validate_input(self.text[:self.cursor_pos] + pasted + self.text[self.cursor_pos:]):
                            self._add_to_history()
                            self._delete_selection()
                            self.text = self.text[:self.cursor_pos] + pasted + self.text[self.cursor_pos:]
                            self.cursor_pos += len(pasted)
                except pygame.error:
                    pass
                return True

        # Navegação
        old_pos = self.cursor_pos
        
        if key == pygame.K_LEFT:
            if ctrl_pressed:
                # Ctrl+Left: pular palavra
                while self.cursor_pos > 0 and not self.text[self.cursor_pos-1].isalnum():
                    self.cursor_pos -= 1
                while self.cursor_pos > 0 and self.text[self.cursor_pos-1].isalnum():
                    self.cursor_pos -= 1
            else:
                if self.cursor_pos > 0:
                    self.cursor_pos -= 1
        elif key == pygame.K_RIGHT:
            if ctrl_pressed:
                # Ctrl+Right: pular palavra
                while self.cursor_pos < len(self.text) and self.text[self.cursor_pos].isalnum():
                    self.cursor_pos += 1
                while self.cursor_pos < len(self.text) and not self.text[self.cursor_pos].isalnum():
                    self.cursor_pos += 1
            else:
                if self.cursor_pos < len(self.text):
                    self.cursor_pos += 1
        elif key == pygame.K_HOME:
            self.cursor_pos = 0
        elif key == pygame.K_END:
            self.cursor_pos = len(self.text)

        # Seleção com Shift
        if shift_pressed and key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_HOME, pygame.K_END]:
            if self.selection_start == -1:
                self.selection_start = old_pos
            self.selection_end = self.cursor_pos
        elif not shift_pressed and key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_HOME, pygame.K_END]:
            self.selection_start = self.selection_end = -1

        # Edição
        if key == pygame.K_BACKSPACE:
            if not self._delete_selection():
                if self.cursor_pos > 0:
                    self._add_to_history()
                    self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                    self.cursor_pos -= 1
        elif key == pygame.K_DELETE:
            if not self._delete_selection():
                if self.cursor_pos < len(self.text):
                    self._add_to_history()
                    self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]
        elif key == pygame.K_RETURN:
            self.active = False
            if self.on_change:
                self.on_change(self.text)
        elif key == pygame.K_ESCAPE:
            self.active = False
        elif key == pygame.K_SPACE:
            if len(self.text) < self.max_length:
                new_text = self.text[:self.cursor_pos] + ' ' + self.text[self.cursor_pos:]
                if self._validate_input(new_text):
                    self._add_to_history()
                    self._delete_selection()
                    self.text = new_text
                    self.cursor_pos += 1
        else:
            # Inserção de caracteres normais
            if not ctrl_pressed:
                char = self._get_char_from_key(key, shift_pressed)
                if char and len(self.text) < self.max_length:
                    new_text = self.text[:self.cursor_pos] + char + self.text[self.cursor_pos:]
                    if self._validate_input(new_text):
                        self._add_to_history()
                        self._delete_selection()
                        self.text = new_text
                        self.cursor_pos += 1

        # Reset cursor blink
        self.cursor_visible = True
        self.cursor_timer = 0
        return True

    def on_text_input(self, text):
        """ entrada de texto Unicode"""
        if not self.active:
            return False
        

        if len(text) == 1 and ord(text) < 32:
            return False
        
        if len(self.text) + len(text) <= self.max_length:
            new_text = self.text[:self.cursor_pos] + text + self.text[self.cursor_pos:]
            if self._validate_input(new_text):
                self._add_to_history()
                self._delete_selection()
                self.text = new_text
                self.cursor_pos += len(text)
                
                # Reset cursor blink
                self.cursor_visible = True
                self.cursor_timer = 0
                return True
        return False

    def set_text(self, text):
        if self._validate_input(text):
            self._add_to_history()
            self.text = self._format_text(text)
            self.cursor_pos = len(self.text)
            self.selection_start = self.selection_end = -1

    def get_text(self):
        return self.text
    
    def set_input_type(self, input_type):
        """tipo de entrada: 'text', 'number', 'email', 'password'"""
        self.input_type = input_type
        
    def set_validator(self, validator_func):

        self.validator = validator_func
        
    def set_formatter(self, formatter_func):
        self.formatter = formatter_func
        
    def set_max_length(self, max_length):
        self.max_length = max_length
        
    def clear(self):
        self._add_to_history()
        self.text = ""
        self.cursor_pos = 0
        self.selection_start = self.selection_end = -1
        
    def select_all(self):
        self.selection_start = 0
        self.selection_end = len(self.text)