import carla
import numpy as np
import pygame

from pywidgets.core.widget import Widget



class CarlaSurface(Widget):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.camera = None
        self.surface = None
    
    def release(self):
        if self.camera is None:
            return
        self.camera.stop()
        self.camera.destroy()

    def get_transform(self):
        return self.camera.get_transform()
    
    def set_trasform(self,transform):
        self.camera.set_transform(transform)

    def render(self, surface):
        if self.surface is not None:
            surface.blit(self.surface, self.bound)

class CarlaImage(CarlaSurface):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        pass
 
    
    def _process_image(self,image):
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove canal alpha
        array = array[:, :, ::-1]  # BGR para RGB
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.surface.get_size() != (self.rect.width, self.rect.height):
            self.surface = pygame.transform.scale(self.surface, (self.rect.width, self.rect.height))
 


class CarlaMask(CarlaSurface):
    def __init__(self, x, y, width, height,classID=24,color=(255, 255, 255)):
        super().__init__(x, y, width, height)
 
        self.classID=classID
        self.color=color
 
    
    def _process_image(self,image):
        image.convert(carla.ColorConverter.Raw)
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        seg = np.reshape(raw, (image.height, image.width, 4))[:, :, 2]  # Canal azul = classe
        
 
        mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        mask[seg == self.classID] = self.color
 
        
        self.surface = pygame.surfarray.make_surface(mask.swapaxes(0, 1))

        if self.surface.get_size() != (self.rect.width, self.rect.height):
            self.surface = pygame.transform.scale(self.surface, (self.rect.width, self.rect.height))
        
 

class CarlaClient(Widget):
    def __init__(self, host='localhost', port=2000,timeout=10.0):
        super().__init__(0, 0, 0, 0)
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.traffic_manager = self.client.get_trafficmanager()
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(self.settings)

        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()


        self.blueprint = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        vehicle_bp = self.blueprint.find('vehicle.tesla.model3')
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_points[0])
        self.surfaces=[]

    
    def get_vehicle(self):
        return self.vehicle

    def set_minim_objects(self):
        self.dissable_objects(carla.CityObjectLabel.Buildings, False)
        self.dissable_objects(carla.CityObjectLabel.Walls, False)
        self.dissable_objects(carla.CityObjectLabel.TrafficLight, False)
        self.dissable_objects(carla.CityObjectLabel.Rider, False)
        self.dissable_objects(carla.CityObjectLabel.Poles, False)
        self.dissable_objects(carla.CityObjectLabel.TrafficSigns, False)
        self.dissable_objects(carla.CityObjectLabel.Vegetation, False)
        self.dissable_objects(carla.CityObjectLabel.Car, False)
        self.dissable_objects(carla.CityObjectLabel.Water, False)
        self.dissable_objects(carla.CityObjectLabel.Dynamic, False)
        self.dissable_objects(carla.CityObjectLabel.Other, False)
        self.dissable_objects(carla.CityObjectLabel.Static, False)
        self.dissable_objects(carla.CityObjectLabel.GuardRail, False)
        self.dissable_objects(carla.CityObjectLabel.Water, False)
        self.dissable_objects(carla.CityObjectLabel.Fences, False)
        self.dissable_objects(carla.CityObjectLabel.Train, False)
        self.dissable_objects(carla.CityObjectLabel.Bus, False)
        self.dissable_objects(carla.CityObjectLabel.Pedestrians, False)
        self.dissable_objects(carla.CityObjectLabel.Motorcycle, False)
        self.dissable_objects(carla.CityObjectLabel.Bicycle, False)
        self.dissable_objects(carla.CityObjectLabel.RailTrack, False)
        self.dissable_objects(carla.CityObjectLabel.Terrain, False)
        self.dissable_objects(carla.CityObjectLabel.Truck, False)
        #self.dissable_objects(carla.CityObjectLabel.Vehicle, False)


    def dissable_objects(self, type,state):
        env_objs = self.world.get_environment_objects(type)
        objects_to_toggle = {}
        for obj in env_objs:
            objects_to_toggle[obj.id] = False
        self.world.enable_environment_objects(objects_to_toggle, state)

    def create_images(self,x,y,width=256, height=256):
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15))
        cam_rgb = self.blueprint.find("sensor.camera.rgb")
        cam_rgb.set_attribute('image_size_x', str(width))
        cam_rgb.set_attribute('image_size_y', str(height))
        cam_rgb.set_attribute('fov', '90')
        surface = CarlaImage(x, y, width, height)
        surface.camera = self.world.spawn_actor(cam_rgb, camera_transform, attach_to=self.vehicle)
        surface.camera.listen(lambda image: surface._process_image(image))
        self.surfaces.append(surface)
        return surface
    
    def create_masks(self,x,y, width=256, height=256,classID=24,color=(255, 255, 255)):
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15))
        cam_rgb = self.blueprint.find("sensor.camera.semantic_segmentation")
        cam_rgb.set_attribute('image_size_x', str(width))
        cam_rgb.set_attribute('image_size_y', str(height))
        cam_rgb.set_attribute('fov', '90')
        surface = CarlaMask(x, y, width, height,classID,color)
        surface.camera = self.world.spawn_actor(cam_rgb, camera_transform, attach_to=self.vehicle)
        surface.camera.listen(lambda image: surface._process_image(image))
        self.surfaces.append(surface)
        return surface
    
    def update(self, dt):
        self.world.tick()
        return super().update(dt)

    def release(self):
        for surface in self.surfaces:
            surface.release()   
        self.vehicle.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        self.world.apply_settings(carla.WorldSettings(synchronous_mode=False))




