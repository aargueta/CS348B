// cameras/realistic.cpp*
#include "stdafx.h"
#include "cameras/realistic.h"
#include "paramset.h"
#include "sampler.h"
#include "montecarlo.h"
#include "filters/box.h"
#include "film/image.h"
#include "samplers/stratified.h"
#include "intersection.h"
#include "renderer.h"

#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

RealisticCamera *CreateRealisticCamera(const ParamSet &params,
        const AnimatedTransform &cam2world, Film *film) {
	   // Extract common camera parameters from \use{ParamSet}
	   float hither = params.FindOneFloat("hither", -1);
	   float yon = params.FindOneFloat("yon", -1);
	   float shutteropen = params.FindOneFloat("shutteropen", -1);
	   float shutterclose = params.FindOneFloat("shutterclose", -1);

	   // Realistic camera-specific parameters
	   string specfile = params.FindOneString("specfile", "");
	   float filmdistance = params.FindOneFloat("filmdistance", 70.0); // about 70 mm default to film
	   float fstop = params.FindOneFloat("aperture_diameter", 1.0);
	   float filmdiag = params.FindOneFloat("filmdiag", 35.0);
	   string autofocusfile = params.FindOneString("af_zones", "");
	   assert(hither != -1 && yon != -1 && shutteropen != -1 &&
	      shutterclose != -1 && filmdistance!= -1);
	   if (specfile == "") {
	       Severe( "No lens spec file supplied!\n" );
	   }
	   return new RealisticCamera(cam2world, hither, yon,
	      shutteropen, shutterclose, filmdistance, fstop,
	      specfile, autofocusfile, filmdiag, film);
}

RealisticCamera::RealisticCamera(const AnimatedTransform &cam2world,
                                 float hither, float yon,
                                 float sopen, float sclose,
                                 float filmdistance, float aperture_diameter_,
                                 const string &specfile,
								 const string &autofocusfile,
                                 float filmdiag,
								 Film *f)
                                 : Camera(cam2world, sopen, sclose, f),
								   ShutterOpen(sopen),
								   ShutterClose(sclose),
								   film(f)
{

	// YOUR CODE HERE -- build and store datastructures representing the given lens
	// and film placement.
	if (specfile.compare("") != 0) {
		ParseCameraSpec(specfile);
	}
	filmDistance = filmdistance;
	filmDiag = filmdiag;
	filmXRes = (float)f->xResolution;
	filmYRes = (float)f->yResolution;
	float filmRatio = (float)(f->yResolution) / (float)(f->xResolution);
	filmXDim = filmdiag / sqrt(pow(filmRatio, 2.f) + 1);
	float localXDim = filmXDim;
	filmYDim = filmdiag * filmRatio / sqrt(pow(filmRatio, 2.f) + 1);
	float localYDim = filmYDim;

	exitPupilRadius = FindExitPupil(stopIndex);

	// If 'autofocusfile' is the empty string, then you should do
	// nothing in any subsequent call to AutoFocus()
	autofocus = false;

	if (autofocusfile.compare("") != 0)  {
		ParseAfZones(autofocusfile);
		autofocus = true;
	}
}

// Parses the lenses in the camera
void RealisticCamera::ParseCameraSpec(const string& filename) {
	ifstream specfile(filename.c_str());
	if (!specfile) {
		fprintf(stderr, "Cannot open file %s\n", filename.c_str());
		exit(-1);
	}

	char line[512];

	while (!specfile.eof()) {
		specfile.getline(line, 512);
		if (line[0] != '\0' && line[0] != '#' &&
			line[0] != ' ' && line[0] != '\t' && line[0] != '\n') {
			lenses.resize(lenses.size() + 1);
			Lens& lens = lenses[lenses.size() - 1];
			sscanf(line, "%f %f %f %f\n", &lens.radius, &lens.thickness, &lens.indexOfRefraction, &lens.aperture);
		}
	}
	float zIntercept = 0.0f;
	stopIndex = 0;
	for (int i = 0; i < lenses.size(); i++) {
		lenses[i].zIntercept = zIntercept;
		if (lenses[i].radius == 0.f) {
			// Stop detected
			stopIndex = i;
			if (i > 0) {
				lenses[i].indexOfRefraction = lenses[i - 1].indexOfRefraction;
			} else {
				lenses[i].indexOfRefraction = 1.0f;
			}
		}
		zIntercept -= lenses[i].thickness;
	}
	printf("Read in %u lenses from %s\n", lenses.size(), filename.c_str());
	logFile = fopen("rayLog.csv", "w");
	logClosed = new bool;
	raysSent = new int;
	*raysSent = 0;
	*logClosed = false;

	rasXMin = new float;
	rasXMax = new float;
	rasYMin = new float;
	rasYMax = new float;
	*rasXMin = 10000.f;
	*rasXMax = -10000.f;
	*rasYMin = 10000.f;
	*rasYMax = -10000.f;
}

float RealisticCamera::FindExitPupil(int stopIndex) {
	Ray ray;
	CameraSample sample;
	sample.imageX = filmXRes / 2.f;
	sample.imageY = filmYRes / 2.f;
	sample.lensU = 0.f;
	sample.lensV = 1.f;
	while (sample.lensV > 0.f) {
		float weight = GenerateRay(sample, &ray, false);
		if (weight > 0.f) {
			return sample.lensV * lenses[0].aperture;
		}
		sample.lensV -= .01f;
	}
	Assert(false);
}


// parses the AF zone file
void RealisticCamera::ParseAfZones(const string& filename)
{
  ifstream specfile(filename.c_str());
   if (!specfile) {
      fprintf(stderr, "Cannot open file %s\n", filename.c_str());
      exit (-1);
   }

   char line[512];

   while (!specfile.eof()) {
      specfile.getline(line, 512);
      if (line[0] != '\0' && line[0] != '#' &&
         line[0] != ' ' && line[0] != '\t' && line[0] != '\n')
      {
		afZones.resize(afZones.size()+1);
		AfZone& zone = afZones[afZones.size()-1];
		sscanf(line, "%f %f %f %f\n", &zone.left, &zone.right, &zone.top, &zone.bottom);
      }
   }

	printf("Read in %u AF zones from %s\n", afZones.size(), filename.c_str());
}

RealisticCamera::~RealisticCamera()
{
#ifdef DEBUG
	fprintf(logFile, "LIMITS, %f, %f, %f, %f\n", *rasXMin, *rasXMax, *rasYMin, *rasYMax);
#endif DEBUG
	//fclose(logFile);
}
float RealisticCamera::GenerateRay(const CameraSample &sample, Ray *ray) const {
	return GenerateRay(sample, ray, true);
}

float RealisticCamera::GenerateRay(const CameraSample &sample, Ray *ray, bool enRandNWeight) const
{
	// YOUR CODE HERE -- make that ray!

	// use sample->imageX and sample->imageY to get raster-space coordinates
	// of the sample point on the film.
	// use sample->lensU and sample->lensV to get a sample position on the lens

	// GenerateRay() should return the weight of the generated ray
	// Transform pixel space to film space

	float xSample = ((sample.imageX / filmXRes) - 0.5f) * filmXDim;
	float ySample = ((sample.imageY / filmYRes) - 0.5f) * filmYDim;
	float filmZ = lenses.back().zIntercept - filmDistance;
	Point pRas(xSample, ySample, filmZ);
	
#ifdef DEBUG
	// DEBUG: Check sample limits
	*rasXMin = min(*rasXMin, sample.imageX);
	*rasXMax = max(*rasXMax, sample.imageX);
	*rasYMin = min(*rasYMin, sample.imageY);
	*rasYMax = max(*rasYMax, sample.imageY);
#endif DEBUG

	// Create point on sample disk (at origin)
	float lensU, lensV;
	if (enRandNWeight) {
		ConcentricSampleDisk(sample.lensU, sample.lensV, &lensU, &lensV);
	} else {
		lensU = sample.lensU;
		lensV = sample.lensV;
	}
	// Shoot at closest lens for exit pupil testing
	// Otherwise shoot at exit pupil
	lensU *= (enRandNWeight) ? exitPupilRadius : lenses.back().radius;
	lensV *= (enRandNWeight) ? exitPupilRadius : lenses.back().radius;
	Point pLens(lensU, lensV, (enRandNWeight) ? lenses[stopIndex].zIntercept : lenses.back().zIntercept);

	vector<Point> debugRayPts;

	// Create ray from two points
	*ray = Ray(pRas, Normalize(pLens - pRas), 0.f);
	Ray oldRay(*ray);
	debugRayPts.push_back(ray->o);

	// Send ray through lenses, from back (film) to front (origin)
	bool rayGood = true;
	for (int i = lenses.size() - 1; i >= 0; i--) {
		if (i > 0) {
			rayGood = PassRayThroughLens(lenses[i], ray, lenses[i - 1].indexOfRefraction);
		} else {
			rayGood = PassRayThroughLens(lenses[i], ray, 1.0f);
		}
		if (!rayGood) {
			break;
		} else {
			debugRayPts.push_back(ray->o);
		}
		if (oldRay.o.z > ray->o.z) {
			printf("Someshit");
		}
		oldRay = *ray;
	}
	if ((*raysSent) % 16 == 0 && debugRayPts.size() > 1 && *raysSent < 10000) {
		fprintf(logFile, "RAY, ");
		for (int i = 0; i < debugRayPts.size(); i++) {
			Point p = debugRayPts[i];
			if (i < debugRayPts.size() - 1) {
				fprintf(logFile, "%f, %f, %f,", p.x, p.y, p.z);
			} else {
				fprintf(logFile, "%f, %f, %f\n", p.x, p.y, p.z);
			}
		}
		fprintf(logFile, "\n");
	}
	if (*raysSent >= 10000 && !(*logClosed)) {
		fclose(logFile);
		printf("File closed...\n");
		*logClosed = true;
	}
	(*raysSent)++;
	ray->time = sample.time;
	//ray->o /= 1000; // Scale from mm to meters
	Ray tempRay(*ray);
	CameraToWorld(tempRay, ray);
	ray->d = Normalize(ray->d);
	if (rayGood) {
		if (enRandNWeight) {
			float areaExitPupil = M_PI * pow(exitPupilRadius, 2.f);
			float zintExitPupil = lenses[stopIndex].zIntercept;
			float distToExitPupilCenter = (Point(0.f, 0.f, zintExitPupil) - pRas).Length();
			float cosExitPupil = (zintExitPupil - filmZ) / distToExitPupilCenter;
			// This is way too dim, basically mapping it to cos^4
			float irr = areaExitPupil / pow((zintExitPupil - filmZ), 2.f) * pow(cosExitPupil, 4.f);
			return irr;
		} else {
			return 1.f;
		}
	} else {
		return 0.0f;
	}
}

bool RealisticCamera::PassRayThroughLens(Lens lens, Ray* ray, float nextIndexOfRefraction) const {

#ifdef DEBUG
	ray->o.y = 0.f;
	ray->d = Normalize(Vector(ray->d.x, 0.f, ray->d.z));

	Assert(nextIndexOfRefraction != 0.f);
#endif DEBUG

	// Check for stop
	if (lens.radius == 0.f) {
		float t = (lens.zIntercept - ray->o.z) / ray->d.z;
		Point stopInt = ray->o + ray->d * t;
		float stopIntLength = (stopInt - Point(0.f, 0.f, lens.zIntercept)).Length();
		if (stopIntLength > (lens.aperture / 2.f)) {
			// Blocked by stop
			Assert(sqrt(pow(stopInt.x, 2.f) + pow(stopInt.y, 2.f)) > lens.aperture / 2.f);
			return false;
		} else {
			// Passes through aperture, don't update ray
			return true;
		}
	}
	
	// Calculate lens spherical center
	Point sphereC = Point(0.f, 0.f, lens.zIntercept - lens.radius); // Negative radii curve away from film, so center is more positive

	float vUSR = (pow(Dot(ray->d, ray->o - sphereC), 2.f) - (ray->o - sphereC).LengthSquared() + pow(lens.radius, 2.f));
	if (vUSR < 0) {
		// Doesn't intersect with lens
		return false;
	}

	// Calculate ray/lens intersection
	float dPlus = -(Dot(ray->d, ray->o - sphereC)) + sqrt(vUSR);
	float dMinus = -(Dot(ray->d, ray->o - sphereC)) - sqrt(vUSR);
	float d = (lens.radius > 0) ? max(dMinus, dPlus) : min(dMinus, dPlus);
	Point lensInt = ray->o + ray->d * d;
	Vector lensNorm = Normalize(lensInt - sphereC);
	lensNorm *= (lensNorm.z > 0.f) ? -1.f : 1.f; // Get normal pointing right way

#ifdef DEBUG
	// DEBUG: Check for actual intersection on circle
	float calcRadius = (lensInt - sphereC).Length();
	Assert(abs(calcRadius - abs(lens.radius)) < .001f);
#endif DEBUG

	// Check for lens aperture blocking
	Vector appInt = (lensInt - Point(0.f, 0.f, lensInt.z));
	float appIntLength = appInt.Length();
	if (appIntLength > (lens.aperture / 2.f)) {
		// Blocked by aperture
		return false;
	}

	// Calculate new ray direction using Snell's law
	float cosThetaI = abs(Dot(ray->d, lensNorm));
	if (abs(cosThetaI - 1.f) < .00001f) {
		// Ray is basically going straight
		return true;
	}
	float incidentAngle = acosf(Dot(Normalize(ray->d), Normalize(lensNorm)));
	float sinThetaI = sin(incidentAngle);
	float indexRatio = nextIndexOfRefraction / lens.indexOfRefraction;
	float nRatio = lens.indexOfRefraction / nextIndexOfRefraction;

	Vector lensTan = Normalize(ray->d + (cosThetaI * lensNorm));
	Vector tVector, tNorm;
	Vector tTan = nRatio * sinThetaI * lensTan;
	if (tTan.LengthSquared() > 1.0f) {
		tTan = Normalize(tTan);
		tVector = tTan;
	} else {
		tNorm = sqrt(1 - tTan.LengthSquared()) * -lensNorm; // Want normal in ray's direction
		tVector = Normalize(tTan + tNorm);
	}

#ifdef DEBUG
	// DEBUG: lensNorm & Tan and tNorm and tTan are ortho
	float lensDot = Dot(lensNorm, lensTan);
	Assert(abs(lensDot) < 0.01f);
	float tDot = Dot(tTan, tNorm);
	Assert(abs(tDot) < 0.01f);
#endif DEBUG

	// Test for TIR
	float refractedAngle = acosf(Dot(Normalize(tVector), Normalize(lensNorm)));
	if (abs(refractedAngle - (M_PI / 2.0f)) < .01f) {
		// TIR has occurred
		return false;
	}

#ifdef DEBUG
	// DEBUG: Snell's law is satisfied
	float theoRefAngle = asin(nRatio * sin(incidentAngle));
	float theoSinRatio = sin(incidentAngle) / sin(theoRefAngle);
	float sinRatio = sin(incidentAngle) / sin(refractedAngle);
	if (abs(incidentAngle - M_PI) > .01f && incidentAngle > .01f) {
		// Make sure to avoid testing 0/0 situations
		Assert(abs(abs(sinRatio) - indexRatio) < .1f);
	} else {
		Assert(true);
	}
	Assert(tVector.z >= -0.1f); // Make sure still pointing the right way
#endif DEBUG

	// Combine data into ray
	*ray = Ray(lensInt, tVector, 0.f, INFINITY);
	return true;
}

void  RealisticCamera::AutoFocus(Renderer * renderer, const Scene * scene, Sample * origSample) {
	// YOUR CODE HERE:
	// The current code shows how to create a new Sampler, and Film cropped to the size of the auto focus zone.
	// It then renders the film, producing rgb values.  You need to:
	//
	// 1. Modify this code so that it can adjust film plane of the camera
	// 2. Use the results of raytracing to evaluate whether the image is in focus
	// 3. Search over the space of film planes to find the best-focused plane.

	if(!autofocus)
		return;

	for (size_t i=0; i<afZones.size(); i++) {
		bool focusDone = false;
		int focusIters = 0;
		float focusDir = 5.f;
		bool focusFineGrain = false;
		vector<float> fDists;
		vector<float> focusVals;

		while (!focusDone) {
			AfZone & zone = afZones[i];

			RNG rng;
			MemoryArena arena;
			Filter * filter = new BoxFilter(.5f, .5f);
			const float crop[] = { zone.left, zone.right, zone.top, zone.bottom };
			ImageFilm sensor(film->xResolution, film->yResolution, filter, crop, "foo.exr", false);
			int xstart, xend, ystart, yend;
			sensor.GetSampleExtent(&xstart, &xend, &ystart, &yend);

			StratifiedSampler sampler(xstart, xend, ystart, yend,
									  16, 16, true, ShutterOpen, ShutterClose);

			// Allocate space for samples and intersections
			int maxSamples = sampler.MaximumSampleCount();
			Sample *samples = origSample->Duplicate(maxSamples);
			RayDifferential *rays = new RayDifferential[maxSamples];
			Spectrum *Ls = new Spectrum[maxSamples];
			Spectrum *Ts = new Spectrum[maxSamples];
			Intersection *isects = new Intersection[maxSamples];

			// Get samples from _Sampler_ and update image
			int sampleCount;
			while ((sampleCount = sampler.GetMoreSamples(samples, rng)) > 0) {
				// Generate camera rays and compute radiance along rays
				for (int i = 0; i < sampleCount; ++i) {
					// Find camera ray for _sample[i]_

					float rayWeight = this->GenerateRayDifferential(samples[i], &rays[i]);
					rays[i].ScaleDifferentials(1.f / sqrtf(sampler.samplesPerPixel));


					// Evaluate radiance along camera ray

					if (rayWeight > 0.f)
						Ls[i] = rayWeight * renderer->Li(scene, rays[i], &samples[i], rng,
						arena, &isects[i], &Ts[i]);
					else {
						Ls[i] = 0.f;
						Ts[i] = 1.f;
					}

					// Issue warning if unexpected radiance value returned
					if (Ls[i].HasNaNs()) {
						Error("Not-a-number radiance value returned "
							  "for image sample.  Setting to black.");
						Ls[i] = Spectrum(0.f);
					} else if (Ls[i].y() < -1e-5) {
						Error("Negative luminance value, %f, returned"
							  "for image sample.  Setting to black.", Ls[i].y());
						Ls[i] = Spectrum(0.f);
					} else if (isinf(Ls[i].y())) {
						Error("Infinite luminance value returned"
							  "for image sample.  Setting to black.");
						Ls[i] = Spectrum(0.f);
					}

				}

				// Report sample results to _Sampler_, add contributions to image
				if (sampler.ReportResults(samples, rays, Ls, isects, sampleCount)) {
					for (int i = 0; i < sampleCount; ++i) {

						sensor.AddSample(samples[i], Ls[i]);

					}
				}

				// Free _MemoryArena_ memory from computing image sample values
				arena.FreeAll();
			}

			float * rgb;
			int width;
			int height;
			sensor.WriteRGB(&rgb, &width, &height, 1.f);
			// YOUR CODE HERE! The rbg contents of the image for this zone
			// are now stored in the array 'rgb'.  You can now do whatever
			// processing you wish
			float* modLap;
			CalcModLaplacian(rgb, &modLap, width, height);
			float currFocusVal = 0;

			std::stringstream modLapFilename;
			modLapFilename << "modlap" << focusIters << ".csv";
			FILE* modLapFile = fopen(modLapFilename.str().c_str(), "w");
			Assert(modLapFile != NULL);
			int mlX = width - 2;
			int mlY = height - 2;
			for (int j = 0; j < mlY; j++) {
				for (int k = 0; k < mlX; k++) {
					currFocusVal += modLap[k + mlX * j];
					if (k < mlX - 1) {
						fprintf(modLapFile, "%f, ", modLap[k + mlX * j]);
					} else {
						fprintf(modLapFile, "%f\n", modLap[k + mlX * j]);
					}
				}
			}
			printf("Focusing, iter #%u = %f\n", focusIters, currFocusVal);
 			fclose(modLapFile);

			// Record current settings
			fDists.push_back(filmDistance);
			focusVals.push_back(currFocusVal);

			// Set next settings
			focusFineGrain = (focusIters >= 2);
			if (!focusFineGrain) {
				// Test 3 times in arbitrary direction
				filmDistance += focusDir;
			} else {
				if (focusIters == 2) {
					// Determine final direction
					if (max(focusVals[0], focusVals[1]) < focusVals[2]) {
						// #2 best, go from #1 -> #2
						focusDir = abs(focusDir / 2.f);
						filmDistance = fDists[1] + focusDir;
					} else if (max(focusVals[1], focusVals[2]) < focusVals[0]) {
						// #0 best, go from #1 -> #0
						focusDir = -abs(focusDir / 2.f);
						filmDistance = fDists[1] + focusDir;
					}else{
						// #1 best, go towards second best
						if (focusVals[0] > focusVals[2]) {
							focusDir = -abs(focusDir / 2.f);
						} else {
							focusDir = abs(focusDir / 2.f);
						}
						filmDistance = fDists[1] + focusDir;
					}
				} else {
					int prevIndex = focusVals.size() - 2;
					if (focusVals[prevIndex] > focusVals.back()) {
						// Went too far, increase search granularity
						focusDir /= 2.f;
						// Undo bad search
						fDists.back() = fDists[prevIndex];
						focusVals.back() = focusVals[prevIndex];
						// Start closer search
						filmDistance = fDists[prevIndex] + focusDir;
						if (abs(focusDir) < 1.f) {
							// Close enough, give up after cleaning up
							focusIters = 12; // Enough to give up
						}
					} else {
						// Getting closer, continue in same direction
						filmDistance += focusDir;
					}
				}
			}

			//you own rgb  now so make sure to delete it:
			delete[] rgb;
			//if you want to see the output rendered from your sensor, uncomment this line (it will write a file called foo.exr)
			sensor.WriteImage(1.f);


			delete[] samples;
			delete[] rays;
			delete[] Ls;
			delete[] Ts;
			delete[] isects;

			if (focusIters > 10) {
				break;
			}
			focusIters++;
		}
	}
}

void RealisticCamera::CalcModLaplacian(float* rgb, float** modLap, int width, int height) {
	int nX = width - 2;
	int nY = height - 2;
	Assert(nX > 0 && nY > 0);
	*modLap = new float[width * height];
	for (int j = 0; j < nY; j++) {
		for (int i = 0; i < nX; i++) {
			int x = i + 1;
			int y = j + 1;
			float iXY = RGBToI(&rgb[3 * (x + y * width)]);
			float imXY = RGBToI(&rgb[3 * ((x - 1) + y * width)]);
			float ipXY = RGBToI(&rgb[3 * ((x + 1) + y * width)]);
			float iXmY = RGBToI(&rgb[3 * (x + (y - 1) * width)]);
			float iXpY = RGBToI(&rgb[3 * (x + (y + 1) * width)]);

			(*modLap)[i + j*nX] = abs(2.f * iXY - imXY - ipXY);
			(*modLap)[i + j*nX] += abs(2.f * iXY - iXmY - iXpY);
		}
	}
}

void RealisticCamera::WriteModLapToRGB(float* modLap, int dimModLap, float** rgb) {
	(*rgb) = new float[3 * dimModLap];
	for (int i = 0; i < dimModLap; i++) {
		IToGrey(modLap[i], &((*rgb)[i * 3]));
	}
}

float RealisticCamera::RGBToI(float* rgb) {
	return (rgb[0] + rgb[1] + rgb[2]) / 3.0f;
}

void RealisticCamera::IToGrey(float i, float* rgb) {
	rgb[0] = rgb[1] = rgb[2] = i;
}