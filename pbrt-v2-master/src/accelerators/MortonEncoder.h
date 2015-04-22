#pragma once

struct MortonIndex {
	MortonIndex() {
		index = 0;
		encoding = 0;
	}

	void Init(uint32_t primIndex, uint32_t mortonEncoding) {
		index = primIndex;
		encoding = mortonEncoding;
	}

	uint32_t index;
	uint32_t encoding;
};
typedef struct MortonIndex MortonIndex;

class MortonEncoder {
public:
	MortonEncoder();
	~MortonEncoder();

	void SetBoundingBox(BBox bounds);
	uint32_t CalculateMortonEncoding(Point point);

	static uint32_t findMortonPartition(vector<MortonIndex*> mortonIndices, uint32_t start, uint32_t end, uint32_t mortonBit);

private:
	float xScale, yScale, zScale;
	float xMin, yMin, zMin;

	uint32_t spreadEvery3(uint32_t a);
	inline uint32_t ScaleTo10b(float value, float min, float range);
};

