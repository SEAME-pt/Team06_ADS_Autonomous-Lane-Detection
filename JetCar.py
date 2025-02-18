import smbus2
import time
import math


class JetCar:
    def __init__(self, servo_addr=0x40, motor_addr=0x60):
        self.servo_bus = smbus2.SMBus(1)
        self.SERVO_ADDR = servo_addr
        self.STEERING_CHANNEL = 0
        self.motor_bus = smbus2.SMBus(1)
        self.MOTOR_ADDR = motor_addr

        self.MAX_ANGLE = 180
        self.SERVO_CENTER_PWM = 320 + 0
        self.SERVO_LEFT_PWM = 320 - 250
        self.SERVO_RIGHT_PWM = 320 + 250
        
        self.is_running = False
        self.current_speed = 0
        self.target_speed = 0
        
        # Inicialização
        self.init_servo()
        self.init_motors()

    def init_servo(self):
        try:
            # Reset PCA9685
            self.servo_bus.write_byte_data(self.SERVO_ADDR, 0x00, 0x06)
            time.sleep(0.1)
            
            # Setup servo control
            self.servo_bus.write_byte_data(self.SERVO_ADDR, 0x00, 0x10)
            time.sleep(0.1)
            
            # Set frequency (~50Hz)
            self.servo_bus.write_byte_data(self.SERVO_ADDR, 0xFE, 0x79)
            time.sleep(0.1)
            
            # Configure MODE2
            self.servo_bus.write_byte_data(self.SERVO_ADDR, 0x01, 0x04)
            time.sleep(0.1)
            
            # Enable auto-increment
            self.servo_bus.write_byte_data(self.SERVO_ADDR, 0x00, 0x20)
            time.sleep(0.1)
            
            return True
        except Exception as e:
            print(f"Servo init error: {e}")
            return False

    def init_motors(self):
        try:
            # Configure motor controller
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0x00, 0x20)
            
            # Set frequency to 60Hz
            prescale = int(math.floor(25000000.0 / 4096.0 / 100 - 1))
            oldmode = self.motor_bus.read_byte_data(self.MOTOR_ADDR, 0x00)
            newmode = (oldmode & 0x7F) | 0x10
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0x00, newmode)
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0xFE, prescale)
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0x00, oldmode)
            time.sleep(0.005)
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0x00, oldmode | 0xa1)
            
            return True
        except Exception as e:
            print(f"Motor init error: {e}")
            return False

   
   

    def set_steering(self, angle):
        angle = max(-self.MAX_ANGLE, min(self.MAX_ANGLE, angle))
        
        if angle < 0:
            pwm = int(self.SERVO_CENTER_PWM + 
                     (angle / self.MAX_ANGLE) * (self.SERVO_CENTER_PWM - self.SERVO_LEFT_PWM))
        elif angle > 0:
            pwm = int(self.SERVO_CENTER_PWM + 
                     (angle / self.MAX_ANGLE) * (self.SERVO_RIGHT_PWM - self.SERVO_CENTER_PWM))
        else:
            pwm = self.SERVO_CENTER_PWM
            
        self.set_servo_pwm(self.STEERING_CHANNEL, 0, pwm)
        self.current_angle = angle

    def set_angle(self, angle):
        self.set_servo_pwm(self.STEERING_CHANNEL, 0, angle)

    def angle_to_pwm(self, angle):
        angle = max(-self.MAX_ANGLE, min(self.MAX_ANGLE, angle))
        
        if angle < 0:  # Esquerda
            # Interpola entre centro e esquerda
            return int(self.SERVO_CENTER_PWM + 
                      (angle / self.MAX_ANGLE) * (self.SERVO_CENTER_PWM - self.SERVO_LEFT_PWM))
        elif angle > 0:  # Direita
            # Interpola entre centro e direita
            return int(self.SERVO_CENTER_PWM + 
                      (angle / self.MAX_ANGLE) * (self.SERVO_RIGHT_PWM - self.SERVO_CENTER_PWM))
        else:  # Centro
            return self.SERVO_CENTER_PWM

    def set_servo_pwm(self, channel, on_value, off_value):
        """Set PWM values for servo"""
        try:
            base_reg = 0x06 + (channel * 4)
            self.servo_bus.write_byte_data(self.SERVO_ADDR, base_reg, on_value & 0xFF)
            self.servo_bus.write_byte_data(self.SERVO_ADDR, base_reg + 1, on_value >> 8)
            self.servo_bus.write_byte_data(self.SERVO_ADDR, base_reg + 2, off_value & 0xFF)
            self.servo_bus.write_byte_data(self.SERVO_ADDR, base_reg + 3, off_value >> 8)
            return True
        except Exception as e:
            print(f"Servo PWM error: {e}")
            return False

    def set_motor_pwm(self, channel, value):
        """Set PWM value for motor channel"""
        value = min(max(value, 0), 4095)
        try:
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0x06 + 4 * channel, 0)
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0x07 + 4 * channel, 0)
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0x08 + 4 * channel, value & 0xFF)
            self.motor_bus.write_byte_data(self.MOTOR_ADDR, 0x09 + 4 * channel, value >> 8)
        except Exception as e:
            print(f"Motor PWM error: {e}")

 

    def set_speed(self, speed):
        speed = max(-100, min(100, speed))
        
        pwm_value = int(abs(speed) / 100.0 * 4095)
        
        if speed > 0:  # Forward
            self.set_motor_pwm(0, pwm_value)  # IN1 Direita
            self.set_motor_pwm(1, 0)          # IN2
            self.set_motor_pwm(2, pwm_value)  # ENA

            self.set_motor_pwm(5, pwm_value)  # IN3 Esquerda
            self.set_motor_pwm(6, 0)          # IN4
            self.set_motor_pwm(7, pwm_value)  # ENB
        elif speed < 0:  # Backward
            self.set_motor_pwm(0, pwm_value)  # IN1 Direita
            self.set_motor_pwm(1, pwm_value)  # IN2
            self.set_motor_pwm(2, 0)          # ENA

         
            self.set_motor_pwm(5, 0)          # IN3
            self.set_motor_pwm(6, pwm_value)  # IN4
            self.set_motor_pwm(7, pwm_value)  # ENB
        else:  # Stop
            for channel in range(9):
                self.set_motor_pwm(channel, 0)
        
        self.current_speed = speed
        
        self.current_speed = speed
   

    def start(self):
        self.running = True
   
    def stop(self):
        self.running = False
        
        self.set_speed(0)
        self.set_steering(0)
        self.servo_bus.close()
        self.motor_bus.close()