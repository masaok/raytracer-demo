#ifndef Objects_h
#define Objects_h

#include "color.h"
#include "vect.h"
#include "ray.h"
#include <cmath>
#include <vector>
#include <iostream>

enum Texture_t {MAT, REFLECTIVE, SPECULAR};
// abstract base class for objects in our scene
class Object {
public:
	Color_t color;
	Texture_t texture;
	Object(Color_t color_, Texture_t texture_= MAT) : color(color_), texture(texture_) {}
	virtual Vec3 get_normal(const Vec3& p) const = 0;
	virtual bool intersect(const Ray& ray, double& t) const = 0;
	const double SELF_AVOID_T = 1e-2;
};

class Sphere : public Object {
    Vec3 Center;
    double Radius;

public:
    Sphere(Vec3 center, double radius, Color_t color, Texture_t texture=MAT) : Center(center), Radius(radius), Object(color, texture) {}
    
    Vec3 get_center() const{
        return Center;
    }

    Vec3 get_normal(const Vec3& p) const {
		return ((p - Center)*(-1/Radius)).normalize();// *(-1 / Radius);
    }

    bool intersect(const Ray& ray, double &t) const {
        Vec3 v = ray.origin - Center;

        const double b = 2 * v.dot(ray.direction);
        const double c = v.dot(v) - Radius*Radius;
        double delta = b*b - 4 * c;

		if (delta < 0) {
			t = FLT_MAX; // no intersection, at 'infinity'
			return false;
		}

        const double t1 = (-b - sqrt(delta))/2;
        const double t2 = (-b + sqrt(delta))/2;

		if (t2 < SELF_AVOID_T) { // the sphere is behind us
			t = FLT_MAX; 
			return false;
		}

		t = (t1 >= SELF_AVOID_T) ? t1 : t2; 
        
        return true;
    }
};


class Cylinder : public Object {
	Vec3 center;
	Vec3 direction;
	double radius;
	double height;

public:
	Cylinder(Vec3 center_, Vec3 direction_, double radius_, double height_, Color_t color, Texture_t texture = MAT) : center(center_), direction(direction_.normalize()), radius(radius_), height(height_), Object(color, texture) {}

	Vec3 get_center() const {
		return center;
	}

	Vec3 get_normal(const Vec3& p) const {
		Vec3 to_center = p - center;
		return ((to_center - direction*(to_center.dot(direction))).normalize());
	}

	bool intersect(const Ray & ray, double& t) const {
		Vec3 rel_origin = ray.origin - center;

		const double directions_dot = ray.direction.dot(direction);
		const double a = 1 - directions_dot* directions_dot;
		const double b = 2 * (rel_origin.dot(ray.direction) - directions_dot *rel_origin.dot(direction));
		const double c = rel_origin.dot(rel_origin) -rel_origin.dot(direction)* rel_origin.dot(direction) - radius * radius;

		double delta = b * b - 4 * a * c;

		if (delta < 0) { 
			t = FLT_MAX; // no intersection, at 'infinity'
			return false;
		}

		const double sqrt_delta_2a = sqrt(delta) / (2 * a);
		double t1 = (-b) / (2*a);
		const double t2 = t1 + sqrt_delta_2a;
		t1 -= sqrt_delta_2a;

		if (t2 < SELF_AVOID_T) { // the cylinder is behind us
			t = FLT_MAX; // no intersection, at 'infinity'
			return false;
		}
		double center_proj = center.dot(direction);
		double t1_proj = ray.get_point(t1).dot(direction);
		if (t1 >= SELF_AVOID_T && t1_proj > center_proj && t1_proj < center_proj+height) {
			t = t1;
			return true;
		}
		double t2_proj = ray.get_point(t2).dot(direction);
		if (t2 >= SELF_AVOID_T && t2_proj > center_proj && t2_proj < center_proj+height) {
			t = t2;
			return true;
		}
		t = FLT_MAX; // no intersection, at 'infinity'
		return false;
	}
	/*
	Circle bottom_circle() {
		return Circle(center, direction, radius, color, texture);
	}
	Circle top_circle() {
		return Circle(center+direction*height, direction, radius, color, texture);
	}
	static void create_capped_cylinder(Scene& scene) {
		// create a cylinder and 2 circles?
	}
	*/
};
// Cone's still not correct.
class Cone : public Object {
	Vec3 center;
	Vec3 direction;
	double slope;
	double height;

public:
	Cone(Vec3 center_, Vec3 direction_, double slope_, double height_, Color_t color, Texture_t texture = MAT) : center(center_), direction(direction_.normalize()), slope(slope_), height(height_), Object(color, texture) {}

	Vec3 get_center() const {
		return center;
	}
	
	Vec3 get_normal(const Vec3& p) const {
		Vec3 to_center = p - center;
		return ((to_center - direction * (to_center.dot(direction) - slope)).normalize());
	}

	bool intersect(const Ray & ray, double& t) const {
		Vec3 rel_origin = ray.origin - center;

		const double directions_dot = ray.direction.dot(direction);
		const double a = 1 - slope*directions_dot * directions_dot;
		const double b = 2 * (rel_origin.dot(ray.direction) - slope*directions_dot * rel_origin.dot(direction));
		const double c = rel_origin.dot(rel_origin) - slope*rel_origin.dot(direction) * rel_origin.dot(direction)-50.0*50.0;

		double delta = b * b - 4 * a * c;

		if (delta < 0) { // was 1e-4, why?
			t = FLT_MAX; // no intersection, at 'infinity'
			return false;
		}

		const double sqrt_delta_2a = sqrt(delta) / (2 * a);
		double t1 = (-b) / (2 * a);
		const double t2 = t1 + sqrt_delta_2a;
		t1 -= sqrt_delta_2a;

		if (t2 < SELF_AVOID_T) { // the cylinder is behind us
			t = FLT_MAX; // no intersection, at 'infinity'
			return false;
		}
		double center_proj = center.dot(direction);
		double t1_proj = ray.get_point(t1).dot(direction);
		if (t1 >= SELF_AVOID_T && t1_proj > center_proj && t1_proj < center_proj + height) {
			t = t1;
			return true;
		}
		double t2_proj = ray.get_point(t2).dot(direction);
		if (t2 >= SELF_AVOID_T && t2_proj > center_proj && t2_proj < center_proj + height) {
			t = t2;
			return true;
		}
		t = FLT_MAX; // no intersection, at 'infinity'
		return false;
	}
};
class Plane : public Object { 
protected:
	Vec3 center;
	Vec3 direction;

public:
	Plane(Vec3 center_, Vec3 direction_, Color_t color, Texture_t texture = MAT) : center(center_), direction(direction_.normalize()), Object(color, texture) {}

	Vec3 get_center() const {
		return center;
	}

	Vec3 get_normal(const Vec3& p) const {
		return direction;
	}

	virtual bool intersect(const Ray& ray, double& t) const {
		double directions_dot_prod = direction.dot(ray.direction);
		if (directions_dot_prod == 0) {// the plane and ray are parallel
			t = FLT_MAX; // no intersection, at 'infinity'
			return false;
		}
		t = direction.dot(center - ray.origin) / directions_dot_prod;

		if (t < SELF_AVOID_T) { // the plane is behind the ray
			t = FLT_MAX;
			return false;
		}

		return true;
	}
};

class Circle : public Plane { 
	double radius;
public:
	Circle(Vec3 center_, Vec3 direction_, double radius_, Color_t color, Texture_t texture = MAT) : radius(radius_), Plane(center_, direction_, color, texture) {}
	
	bool intersect(const Ray & ray, double& t) const {
		if (!Plane::intersect(ray, t)) { // the ray doesnt even hit the plane
			return false;
		}
		Vec3 intersect_point = ray.get_point(t);
		
		if ((intersect_point - center).norm2() > radius*radius) { // intersects with plane outside circle
			t = FLT_MAX; 
			return false;
		}
		
		return true;
	}
};
#endif

