#pragma once
#ifndef Color_h
#define Color_h

class Color_t  {

public:
	int r, g, b;
	Color_t(int red, int green, int blue) : r(red), g(green), b(blue) {}
	
	Color_t scale_by(double scalar)const {
		return (scalar > 0) ? Color_trunc(scalar * r, scalar * g, scalar * b) : Color_t(0,0,0);
	}
	Color_t scale_by2(double scalar)const {
		if (scalar < 0)
			return  Color_t(0, 0, 0);
		scalar *= scalar;
		return Color_trunc(scalar * r, scalar * g, scalar * b);
	}
	Color_t mix_with(const Color_t& rhs) const {
		return Color_t(r * rhs.r, g * rhs.g, b * rhs.b);
	}
	Color_t operator *(double scalar)const {
		return Color_t(scalar * r, scalar * g, scalar * b);
	}
	Color_t operator +(const Color_t& rhs) const {
		return Color_trunc(r + rhs.r, g + rhs.g, b + rhs.b);
	}
	static int trunc(int val) {
		return (val > 255) ? 255 : val;
	}
	static Color_t Color_trunc(int red, int green, int blue) {
		return Color_t(trunc(red), trunc(green), trunc(blue));
	}
};

#endif
