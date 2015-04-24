#include "StdAfx.h"
#include "MortonEncoder.h"

MortonEncoder::MortonEncoder() {
}


MortonEncoder::~MortonEncoder() {
}


void MortonEncoder::SetBoundingBox(BBox bounds) {
	xMin = bounds.pMin.x;
	yMin = bounds.pMin.y;
	zMin = bounds.pMin.z;
	xScale = bounds.pMax.x - bounds.pMin.x;
	yScale = bounds.pMax.y - bounds.pMin.y;
	zScale = bounds.pMax.z - bounds.pMin.z;
}

uint32_t MortonEncoder::CalculateMortonEncoding(Point point) {
	uint32_t x = ScaleTo10b(point.x, xMin, xScale);
	uint32_t y = ScaleTo10b(point.y, yMin, yScale);
	uint32_t z = ScaleTo10b(point.z, zMin, zScale);
	return (spreadEvery3(x) | spreadEvery3(y) << 1 | spreadEvery3(z) << 2);
}


uint32_t MortonEncoder::findMortonPartition(vector<MortonIndex*> mortonIndices, uint32_t start, uint32_t end, uint32_t mortonBit) {
	uint32_t bitMask = 1 << mortonBit;
	if ((mortonIndices[start]->encoding & bitMask) > 0) {
		return start;
	}
	
	uint32_t dist = end - start;
	if (dist == 0) {
		return start;
	}

	uint32_t center = start + (dist / 2);
	if ((mortonIndices[center]->encoding & bitMask) > 0) {
		return findMortonPartition(mortonIndices, center, end, mortonBit);
	} else {
		return findMortonPartition(mortonIndices, start, center, mortonBit);
	}
}

uint32_t MortonEncoder::spreadEvery3(uint32_t a) {
	// Mask only 10-bits
	uint32_t b = a & 0x000003ff;
	b = (b | b << 18) & 0xe00007f; // Move 1st group of 3 up
	b = (b | b << 12) & 0xe07000f; // Move 2nd group of 3 up
	b = (b | b << 06) & 0xe070381; // Move 3rd group of 3 up
	b = (b | b >> 02) & 0x984c261; // Move last 2 bits off groups
	b = (b | b >> 02) & 0x9249249; // Move last bit off of groups
	return b;
}

inline uint32_t MortonEncoder::ScaleTo10b(float value, float min, float range) {
	return Clamp((uint32_t)((value - min) / range * 1023), 0, 1023);
}