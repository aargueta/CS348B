#if defined(_MSC_VER)
#pragma once
#endif

#ifndef PBRT_CAMERAS_REALISTIC_H
#define PBRT_CAMERAS_REALISTIC_H

#include "pbrt.h"
#include "camera.h"
#include "film.h"
#include <fstream>
#include <iterator>

class Lens {
public:
	float radius;
	float thickness;
	float zIntercept;
	float indexOfRefraction;
	float aperture;
};

// Example representation of an autofocus zone.
class AfZone {
	public:
	  // from the data file
	  float left, right;
	  float top, bottom;
	  int xres,yres;
};

class RealisticCamera : public Camera {
public:
   RealisticCamera(const AnimatedTransform &cam2world,
      float hither, float yon, float sopen,
      float sclose, float filmdistance, float aperture_diameter,
      const string &specfile,
	  const string &autofocusfile,
      float filmdiag,
	  Film *film);
   ~RealisticCamera();
   float FindExitPupil(int stopIndex);
   float GenerateRay(const CameraSample &sample, Ray * ray) const;
   float GenerateRay(const CameraSample &sample, Ray * ray, bool enRandNWeight = true) const;
   void  AutoFocus(Renderer * renderer, const Scene * scene, Sample * origSample);
   void  ParseCameraSpec(const string& filename);
   void  ParseAfZones(const string& filename);
   bool  PassRayThroughLens(Lens lens, Ray* ray, float prevIndexOfRefraction) const;

private:
   bool  autofocus;
   vector<Lens> lenses;
   int stopIndex;
   float filmDistance;
   float filmDiag;
   float filmXDim, filmYDim;
   float filmXRes, filmYRes;
   float exitPupilRadius;
   vector<AfZone> afZones;
   float ShutterOpen;
   float ShutterClose;
   Film * film;

   FILE* logFile;
   bool* logClosed;
   int* raysSent;

   float* rasXMin;
   float* rasXMax;
   float* rasYMin;
   float* rasYMax;

};

RealisticCamera *CreateRealisticCamera(const ParamSet &params,
        const AnimatedTransform &cam2world, Film *film);

#endif
