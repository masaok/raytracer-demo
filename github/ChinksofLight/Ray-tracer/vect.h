#ifndef Vect_h
#define Vect_h

#include <cmath>

using namespace std;

class Vec3 {
public:

    double X;
    double Y;
    double Z;

    Vec3(double x, double y, double z) : X(x), Y(y), Z(z) {}

    Vec3 operator +(const Vec3& vec) const{
        return Vec3(X+vec.X,Y+vec.Y,Z+vec.Z);
    }

    Vec3 operator -(const Vec3& vec) const {
        return Vec3(X-vec.X,Y-vec.Y,Z-vec.Z);
    }

    Vec3 operator *(double scalar) const{
        return Vec3(scalar*X,scalar*Y,scalar*Z);
    }

	// never actually used (so far)
    Vec3 directProduct(const Vec3& vec) const{
        return Vec3(vec.X*X,vec.Y*Y,vec.Z*Z);
    }

	Vec3 crossProduct(const Vec3& vec) const {
		return Vec3(vec.Y*Z - vec.Z*Y, vec.Z*X - vec.X*Z, vec.X*Y - vec.Y*X);
	}

    double dot(const Vec3& vec) const {
        return X*vec.X+Y*vec.Y+Z*vec.Z;
    }

    double norm() const {
        return sqrt(dot(*this));
    }
	double norm2() const {
		return dot(*this);
	}
    
    Vec3 normalize() {
        return (*this)*(1/norm());
    }
    
};
#endif
