#ifndef Scene_h
#define Scene_h

#include "objects.h"
#include "color.h"
#include <iostream>

#include <list>


using namespace std;

class Lightsource {
public:
	Vec3 position;
	Color_t color;
	double intensity = 100;
	Lightsource(Vec3 position_, Color_t color_ = Color_t(255,255,255), double intensity_=100.0) : position(position_), color(color_), intensity(intensity_) {}
};

class Scene {
   
    list<Object*> objects;
	list<Lightsource> lightsources;
	//Camera camera;

public:
    Scene() {}

    void add(Object *object) {
		objects.push_back(object);
    }
	
	void add(Lightsource light) {
		lightsources.push_back(light);
	}

    Color_t Shading(const Ray& ray, const Object& object, double t, int depth) {          
            Vec3 intersect_point = ray.origin + ray.direction * t;
            Vec3 normal = object.get_normal(intersect_point);
    
			switch (object.texture){ 
			case MAT:
				//This was chinkOfLight's formula
				return (object.color).scale_by(normal.dot(ray.direction) * 0.5);
			case REFLECTIVE:
			{
				Color_t c = (object.color).scale_by(normal.dot(ray.direction) * 0.5); 
				if (depth > 0)
					c = c+ trace_ray(Ray(intersect_point, (ray.direction - normal * ray.direction.dot(normal) * 2).normalize()), &object, depth -1);
				return c;
			}
			case SPECULAR:
			{
				Color_t c = Color_t(0, 0, 0);
				for (const Lightsource& light: lightsources){
					Vec3 light2pos = light.position - intersect_point;
					//specular:
					if (check_occlusion(intersect_point, light.position)) {
						c = c + light.color.scale_by2(ray.reflect_by(normal).dot(light2pos.normalize())); // add check for occlusion
						//mat:
						c = c + (object.color).mix_with(light.color).scale_by(light.intensity / (light.position - intersect_point).norm2());
					}
				}
				//reflections:
				if (depth > 0)
					c = c + trace_ray(Ray(intersect_point, (ray.direction - normal * ray.direction.dot(normal) * 2).normalize()), &object, depth - 1);
				return c;
				
			}
			default:
				throw exception("unrecognized texture");// UNRECOGNIZED_TEXTURE
			}
    }

	Color_t trace(int x, int y) {
// This function works as the camera, translating pixels to rays
		Vec3 ray_origin = Vec3(0, 0, -1000);
		Vec3 ray_direction = Vec3(x, y, 1250).normalize();

		return trace_ray(Ray(ray_origin, ray_direction), 0, 50);
	}
	Color_t trace_ray(const Ray& ray, const Object* exclude_obj, int depth) {
		double min_t = FLT_MAX;
		int min_i = -1;
		const Object* nearest_obj = nullptr;

		double t = FLT_MAX;
		for (const Object* object : objects) {
				if ((*object).intersect(ray, t)) {
					if (min_t > t) {
						nearest_obj = object;
			
						min_t = t;
					}
				}
		}
		if (nearest_obj != nullptr) {
			return Shading(ray, *nearest_obj, min_t, depth);
		}
		return Color_t(0, 0, 0);
	}
	
	bool check_occlusion(Vec3 target, Vec3 source) {
		Vec3 toSource = source - target;
		double t_light = toSource.norm();
		Ray ray = Ray(target, toSource * (1.0 / t_light));
		double min_t = t_light;
		const Object* nearest_obj = nullptr;
		double t = FLT_MAX;
		for (const Object* object : objects) {		
			if ((*object).intersect(ray, t)) {
				if (min_t > t) {
					nearest_obj = object;
					min_t = t;
				}
			}
		}
		return nearest_obj == nullptr; // false if lightsource is occluded
	}
};


#endif