
from pywidgets.core.widget import Widget
from pywidgets.core.skin import *
from pywidgets.widgets.scroll import Scrollbar

import pygame
import math
 

class TextView(Widget):
    def __init__(self, x, y, width, height, text="", editable=False, wrap=True):
        super().__init__(x, y, width, height)
        self.text = text
        self.editable = editable
        self.wrap = wrap
        self.scroll_y = 0
        self.scrollbar = Scrollbar(0, 0, height, vertical=True,
                                   content_size=100, view_size=height,
                                   scroll_pos=0, on_scroll=self._on_scroll)
        self.scrollbar.parent = self
        self.lines = []
        self.cursor_pos = 0  
        self.cursor_line = 0  
        self.cursor_col = 0    
        self.cursor_visible = True
        self.cursor_timer = 0.0
        self.cursor_blink_time = 0.5
        self.focus = False
        self.build = False
        self.clip = pygame.Rect(self.get_x(), self.get_y(), self.rect.width, self.rect.height)
        
     
        self.selection_start = -1
        self.selection_end = -1
        self.mouse_selecting = False
        self.double_click_time = 0.3
        self.last_click_time = 0
        
 
        self.history = [text]
        self.history_index = 0
        self.max_history = 50

    def _on_scroll(self, value):
        self.scroll_y = int(value)

    def _rebuild_lines(self):
        font = self.skin.font
        if self.wrap:
            self.lines = []
            self.line_char_counts = []  # Para rastrear quantos chars cada linha tem
            char_count = 0
            
            for para in self.text.splitlines():
                if not para:  # Linha vazia
                    self.lines.append("")
                    self.line_char_counts.append(0)
                    char_count += 1  # Para o \n
                    continue
                    
                words = para.split(' ')
                line = ""
                line_start_char = char_count
                
                for i, word in enumerate(words):
                    test_line = line + (" " if line else "") + word
                    if font.size(test_line)[0] > self.rect.width - 32:  # Espaço para scrollbar
                        if line:
                            self.lines.append(line)
                            self.line_char_counts.append(len(line))
                            char_count += len(line)
                            line = word
                        else:
                            self.lines.append(word)
                            self.line_char_counts.append(len(word))
                            char_count += len(word)
                            line = ""
                    else:
                        line = test_line
                
                if line:
                    self.lines.append(line)
                    self.line_char_counts.append(len(line))
                    char_count += len(line)
                
                char_count += 1  # Para o \n entre parágrafos
        else:
            self.lines = self.text.splitlines()
            self.line_char_counts = [len(line) for line in self.lines]
        
        self.scrollbar.content_size = len(self.lines) * font.get_height()
        self._update_cursor_position()

    def _update_cursor_position(self):
        if not self.lines:
            self.cursor_line = 0
            self.cursor_col = 0
            return
            
        char_count = 0
        for line_idx, line in enumerate(self.lines):
            if char_count + len(line) >= self.cursor_pos:
                self.cursor_line = line_idx
                self.cursor_col = self.cursor_pos - char_count
                break
            char_count += len(line) + 1  # +1 para \n
        else:
            self.cursor_line = len(self.lines) - 1
            self.cursor_col = len(self.lines[-1]) if self.lines else 0

    def _get_char_pos_from_line_col(self, line, col):
        char_pos = 0
        for i in range(min(line, len(self.lines))):
            char_pos += len(self.lines[i]) + 1  # +1 para \n
        char_pos += min(col, len(self.lines[line]) if line < len(self.lines) else 0)
        return char_pos

    def _get_line_col_from_mouse(self, mouse_x, mouse_y):
        """Converte posição do mouse para linha/coluna"""
        font = self.skin.font
        line_height = font.get_height()
        

        relative_y = mouse_y - self.get_y() - 4 + self.scroll_y
        line = max(0, min(relative_y // line_height, len(self.lines) - 1))

        if line >= len(self.lines):
            return len(self.lines) - 1, 0
            
        relative_x = mouse_x - self.get_x() - 4
        line_text = self.lines[line]
        col = 0
        
        for i, char in enumerate(line_text):
            char_width = font.size(line_text[:i+1])[0]
            if char_width > relative_x:
                break
            col = i + 1
            
        return line, min(col, len(line_text))

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
            self._rebuild_lines()

    def _redo(self):

        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.text = self.history[self.history_index]
            self._rebuild_lines()

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

    def update(self, dt):
        super().update(dt)
        self.scrollbar.update(dt)
        

        self.cursor_timer += dt
        if self.cursor_timer > self.cursor_blink_time:
            self.cursor_timer = 0
            self.cursor_visible = not self.cursor_visible

    def render(self, surface):
        if not self.build:
            self._rebuild_lines()
            self.build = True
        
            
        self.scrollbar.skin = self.skin
        self.scrollbar.set_position(self.bound.width - 16, 0)
        
        font = self.skin.font
        text_color = self.skin.get_property(SkinProps.TEXT_COLOR)
        bg_color = self.skin.get_property(SkinProps.BACKGROUND_COLOR)
        border_color = self.skin.get_property(SkinProps.BORDER_COLOR)
        selection_color = (100, 150, 255, 100)  # Azul semi-transparente

        # Fundo e borda
        pygame.draw.rect(surface, bg_color, self.bound)
        pygame.draw.rect(surface, border_color, self.bound, 1)
        
        # Configurar clipping
        self.clip.x = self.get_x()
        self.clip.y = self.get_y()
        self.clip.width = self.bound.width - 16  # Espaço para scrollbar
        self.clip.height = self.rect.height

        clip = surface.get_clip()
        surface.set_clip(self.clip)

        x = self.get_x() + 4
        y = self.get_y() + 4 - self.scroll_y
        line_height = font.get_height()

        # Renderizar linhas
        for line_idx, line in enumerate(self.lines):
            if y + line_height > self.get_y() and y < self.get_y() + self.rect.height:
                # Renderizar seleção se existir
                if (self.selection_start != -1 and self.selection_end != -1 and 
                    line_idx == self.cursor_line):
                    start_col = min(self.selection_start, self.selection_end) - self._get_char_pos_from_line_col(line_idx, 0)
                    end_col = max(self.selection_start, self.selection_end) - self._get_char_pos_from_line_col(line_idx, 0)
                    
                    if start_col < len(line) and end_col > 0:
                        start_col = max(0, start_col)
                        end_col = min(len(line), end_col)
                        
                        if start_col < end_col:
                            sel_x = x + font.size(line[:start_col])[0]
                            sel_width = font.size(line[start_col:end_col])[0]
                            sel_rect = pygame.Rect(sel_x, y, sel_width, line_height)
                            
                            # Criar surface temporária para transparência
                            temp_surface = pygame.Surface((sel_width, line_height), pygame.SRCALPHA)
                            temp_surface.fill(selection_color)
                            surface.blit(temp_surface, (sel_x, y))
                
                #  texto
                surface.blit(font.render(line, True, text_color), (x, y))
                
                #  cursor se estiver nesta linha
                if (self.editable and self.focus and self.cursor_visible and 
                    line_idx == self.cursor_line):
                    cursor_x = x + font.size(line[:self.cursor_col])[0]
                    pygame.draw.line(surface, text_color, 
                                   (cursor_x, y), (cursor_x, y + line_height), 1)
            
            y += line_height
            if y > self.get_y() + self.rect.height:
                break

        surface.set_clip(clip)
        self.scrollbar.render(surface)

    def on_mouse_down(self, x, y):
        current_time = pygame.time.get_ticks() / 1000.0
        
        if self.scrollbar.bound.collidepoint(x, y):
            return self.scrollbar.on_mouse_down(x, y)
        
        if self.bound.collidepoint(x, y):
            self.focus = True
            
            # Duplo clique para selecionar palavra
            if current_time - self.last_click_time < self.double_click_time:
                self._select_word_at_cursor()
            else:
                line, col = self._get_line_col_from_mouse(x, y)
                self.cursor_line = line
                self.cursor_col = col
                self.cursor_pos = self._get_char_pos_from_line_col(line, col)
                self.selection_start = self.selection_end = -1
                self.mouse_selecting = True
            
            self.last_click_time = current_time
            return True
        else:
            self.focus = False
            return False

    def on_mouse_up(self, x, y):
        self.mouse_selecting = False
        return self.scrollbar.on_mouse_up(x, y)

    def on_mouse_move(self, x, y):
        if self.scrollbar.on_mouse_move(x, y):
            return True
            
        if self.mouse_selecting and self.bound.collidepoint(x, y):
            line, col = self._get_line_col_from_mouse(x, y)
            new_pos = self._get_char_pos_from_line_col(line, col)
            
            if self.selection_start == -1:
                self.selection_start = self.cursor_pos
            self.selection_end = new_pos
            
            self.cursor_line = line
            self.cursor_col = col
            self.cursor_pos = new_pos
            return True
            
        return False

    def on_mouse_wheel(self, delta):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if self.scrollbar.bound.collidepoint(mouse_x, mouse_y):
            return self.scrollbar.on_mouse_wheel(delta)
        if self.bound.collidepoint(mouse_x, mouse_y):
            self.scroll_y = max(0, min(self.scroll_y - delta * 20, 
                                     self.scrollbar.content_size - self.rect.height))
            self.scrollbar.scroll_pos = self.scroll_y
            return True
        return False

    def on_key_down(self, key):
        if not self.editable or not self.focus:
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
                    #   cópia para clipboard
                    pass
                return True
            elif key == pygame.K_x:
                selected = self._get_selected_text()
                if selected:
                    #  corte para clipboard
                    self._add_to_history()
                    self._delete_selection()
                    self._rebuild_lines()
                return True
            elif key == pygame.K_v:
                #  colar do clipboard
                return True

        # Navigation keys
        old_pos = self.cursor_pos
        
        if key == pygame.K_LEFT:
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
        elif key == pygame.K_RIGHT:
            if self.cursor_pos < len(self.text):
                self.cursor_pos += 1
        elif key == pygame.K_UP:
            if self.cursor_line > 0:
                new_line = self.cursor_line - 1
                max_col = len(self.lines[new_line]) if new_line < len(self.lines) else 0
                new_col = min(self.cursor_col, max_col)
                self.cursor_pos = self._get_char_pos_from_line_col(new_line, new_col)
        elif key == pygame.K_DOWN:
            if self.cursor_line < len(self.lines) - 1:
                new_line = self.cursor_line + 1
                max_col = len(self.lines[new_line]) if new_line < len(self.lines) else 0
                new_col = min(self.cursor_col, max_col)
                self.cursor_pos = self._get_char_pos_from_line_col(new_line, new_col)
        elif key == pygame.K_HOME:
            # Ir para início da linha
            self.cursor_pos = self._get_char_pos_from_line_col(self.cursor_line, 0)
        elif key == pygame.K_END:
            # Ir para fim da linha
            if self.cursor_line < len(self.lines):
                self.cursor_pos = self._get_char_pos_from_line_col(self.cursor_line, len(self.lines[self.cursor_line]))

        # Seleção com Shift
        if shift_pressed and key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN, pygame.K_HOME, pygame.K_END]:
            if self.selection_start == -1:
                self.selection_start = old_pos
            self.selection_end = self.cursor_pos
        elif not shift_pressed and key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN, pygame.K_HOME, pygame.K_END]:
            self.selection_start = self.selection_end = -1

        # Edição de texto
        if key == pygame.K_BACKSPACE:
            if not self._delete_selection():
                if self.cursor_pos > 0:
                    self._add_to_history()
                    self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                    self.cursor_pos -= 1
            self._rebuild_lines()
        elif key == pygame.K_DELETE:
            if not self._delete_selection():
                if self.cursor_pos < len(self.text):
                    self._add_to_history()
                    self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]
            self._rebuild_lines()
        elif key == pygame.K_RETURN:
            self._add_to_history()
            self._delete_selection()
            self.text = self.text[:self.cursor_pos] + "\n" + self.text[self.cursor_pos:]
            self.cursor_pos += 1
            self._rebuild_lines()
        elif key == pygame.K_TAB:
            self._add_to_history()
            self._delete_selection()
            self.text = self.text[:self.cursor_pos] + "    " + self.text[self.cursor_pos:]
            self.cursor_pos += 4
            self._rebuild_lines()
        elif key == pygame.K_SPACE:
            # Tratar espaço explicitamente
            self._add_to_history()
            self._delete_selection()
            self.text = self.text[:self.cursor_pos] + " " + self.text[self.cursor_pos:]
            self.cursor_pos += 1
            self._rebuild_lines()
        else:
            # Inserção de texto para outras teclas
            if not ctrl_pressed:  # Evitar inserir texto com Ctrl down
                char = self._get_char_from_key(key, shift_pressed)
                if char:
                    self._add_to_history()
                    self._delete_selection()
                    self.text = self.text[:self.cursor_pos] + char + self.text[self.cursor_pos:]
                    self.cursor_pos += 1
                    self._rebuild_lines()

        # Semre atualizar posição do cursor
        self._update_cursor_position()
        return True

    def _get_char_from_key(self, key, shift_pressed):
        """Converte código de tecla para caractere"""
        
        # Letras (a-z)
        if pygame.K_a <= key <= pygame.K_z:
            char = chr(key)
            return char.upper() if shift_pressed else char
        
        # Números no teclado principal (0-9)
        if pygame.K_0 <= key <= pygame.K_9:
            if shift_pressed:
                # Símbolos com Shift nos números
                shift_numbers = {
                    pygame.K_0: ')', pygame.K_1: '!', pygame.K_2: '@', pygame.K_3: '#', 
                    pygame.K_4: '$', pygame.K_5: '%', pygame.K_6: '^', pygame.K_7: '&', 
                    pygame.K_8: '*', pygame.K_9: '('
                }
                return shift_numbers.get(key, str(key - pygame.K_0))
            else:
                return str(key - pygame.K_0)
        
        # Números do teclado numérico (keypad)
        if pygame.K_KP0 <= key <= pygame.K_KP9:
            return str(key - pygame.K_KP0)
        
        # Símbolos e pontuação
        symbol_map = {
            # Pontuação básica
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
            
            # Teclado numérico
            pygame.K_KP_PERIOD: '.',
            pygame.K_KP_DIVIDE: '/',
            pygame.K_KP_MULTIPLY: '*',
            pygame.K_KP_MINUS: '-',
            pygame.K_KP_PLUS: '+',
            pygame.K_KP_EQUALS: '=',
            
     
            pygame.K_LESS: '<',
            pygame.K_GREATER: '>',
            pygame.K_QUESTION: '?',
            pygame.K_AT: '@',
            pygame.K_CARET: '^',
            pygame.K_AMPERSAND: '&',
            pygame.K_ASTERISK: '*',
            pygame.K_LEFTPAREN: '(',
            pygame.K_RIGHTPAREN: ')',
            pygame.K_UNDERSCORE: '_',
            pygame.K_PLUS: '+',
            pygame.K_COLON: ':',
            pygame.K_QUOTEDBL: '"',
            pygame.K_HASH: '#',
            pygame.K_DOLLAR: '$',
            pygame.K_PERCENT: '%',
            pygame.K_EXCLAIM: '!',
        }
        
 
        if key in symbol_map:
            return symbol_map[key]
        
        # Teclas especiais que não devem inserir caracteres
        special_keys = {
            pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_TAB, pygame.K_BACKSPACE, 
            pygame.K_DELETE, pygame.K_INSERT, pygame.K_HOME, pygame.K_END, 
            pygame.K_PAGEUP, pygame.K_PAGEDOWN, pygame.K_UP, pygame.K_DOWN, 
            pygame.K_LEFT, pygame.K_RIGHT, pygame.K_LSHIFT, pygame.K_RSHIFT, 
            pygame.K_LCTRL, pygame.K_RCTRL, pygame.K_LALT, pygame.K_RALT, 
            pygame.K_LGUI, pygame.K_RGUI, pygame.K_MENU, pygame.K_CAPSLOCK, 
            pygame.K_NUMLOCK, pygame.K_SCROLLOCK, pygame.K_ESCAPE, 
            pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5, 
            pygame.K_F6, pygame.K_F7, pygame.K_F8, pygame.K_F9, pygame.K_F10, 
            pygame.K_F11, pygame.K_F12, pygame.K_F13, pygame.K_F14, pygame.K_F15,
            pygame.K_PRINT, pygame.K_SYSREQ, pygame.K_PAUSE, pygame.K_BREAK
        }
        
        if key in special_keys:
            return None
 
        try:
            key_name = pygame.key.name(key)
            if len(key_name) == 1 and key_name.isprintable():
                return key_name.upper() if shift_pressed else key_name
        except:
            pass
        
 
        return None

    def on_text_input(self, text):
 
        if not self.editable or not self.focus:
            return False
        
        # Filtrar caracteres de controle
        if len(text) == 1 and ord(text) < 32:
            return False
            
        self._add_to_history()
        self._delete_selection()
        
        self.text = self.text[:self.cursor_pos] + text + self.text[self.cursor_pos:]
        self.cursor_pos += len(text)
        self._rebuild_lines()
        self._update_cursor_position()
        
        # Reset cursor blink
        self.cursor_visible = True
        self.cursor_timer = 0
        return True

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

    def set_text(self, text):
        self.text = text
        self.cursor_pos = min(self.cursor_pos, len(text))
        self.selection_start = self.selection_end = -1
        self._add_to_history()
        self._rebuild_lines()

    def get_text(self):
        return self.text
        
    def insert_text(self, text, pos=None):
 
        if pos is None:
            pos = self.cursor_pos
        self._add_to_history()
        self.text = self.text[:pos] + text + self.text[pos:]
        self.cursor_pos = pos + len(text)
        self._rebuild_lines()
        
    def clear(self):
  
        self._add_to_history()
        self.text = ""
        self.cursor_pos = 0
        self.selection_start = self.selection_end = -1
        self._rebuild_lines(),
 
        
        return None

    def on_text_input(self, text):
   
        if not self.editable or not self.focus:
            return False
            
        self._add_to_history()
        self._delete_selection()
        
        self.text = self.text[:self.cursor_pos] + text + self.text[self.cursor_pos:]
        self.cursor_pos += len(text)
        self._rebuild_lines()
        return True

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

    def set_text(self, text):
        self.text = text
        self.cursor_pos = min(self.cursor_pos, len(text))
        self.selection_start = self.selection_end = -1
        self._add_to_history()
        self._rebuild_lines()

    def get_text(self):
        return self.text
        
    def insert_text(self, text, pos=None):
 
        if pos is None:
            pos = self.cursor_pos
        self._add_to_history()
        self.text = self.text[:pos] + text + self.text[pos:]
        self.cursor_pos = pos + len(text)
        self._rebuild_lines()
        
    def clear(self):
     
        self._add_to_history()
        self.text = ""
        self.cursor_pos = 0
        self.selection_start = self.selection_end = -1
        self._rebuild_lines()