#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include<bits/stdc++.h>
#include<cuda.h>

using namespace std;
//m is the fuzzy partition matrix exponent which should be larger than 1 for controlling the degree of fuzzy overlap
#define ull unsigned long long
typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255

static PPMImage *readPPM(const char *filename){
         char buff[16];
         PPMImage *img;
         FILE *fp;
         int c, rgb_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         //read image format
         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n')
    ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}
void writePPM(const char *filename, PPMImage *img){
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}

void FuzzyS(PPMPixel* image, int width, int height, double*** clustMeansPtr, int* pixClust, int s, int numSupPix, int maxIter, double mpol,double m,double M){
	int i, x, y, j;
	x = round((double)s/2);
	y = round((double)s/2);
	double** clusters = (double**)malloc(sizeof(double*)*numSupPix);
	// Cluster Initialisation
	for( i = 0;i<numSupPix;i++){
		int p,q,minX,minY,minGrad = 1e8;
		for(p = -1;p<=1;p++){
			for(q = -1;q<=1;q++){
				if(x+p<1 || x+p >=width-1 || y+q<1 || y+q>=height-1)
					continue;
				int grad = pow(image[(x+p+1)*width + y+q].red -  image[(x+p-1)*width + y+q].red,2) + pow(image[(x+p+1)*width + y+q].green -  image[(x+p-1)*width + y+q].green,2) + pow(image[(x+p+1)*width + y+q].blue -  image[(x+p-1)*width + y+q].blue,2) + pow(image[(x+p)*width + y+q+1].red -  image[(x+p)*width + y+q-1].red,2) + pow(image[(x+p)*width + y+q+1].green -  image[(x+p)*width + y+q-1].green,2) + pow(image[(x+p)*width + y+q+1].blue -  image[(x+p)*width + y+q-1].blue,2);
				if(grad <= minGrad){
					minX = x+p;
					minY = y+q;
					minGrad = grad;
				}
			}
		}
		clusters[i] = (double*)malloc(sizeof(double)*5);
		clusters[i][0] = minX;
		clusters[i][1] = minY;
		clusters[i][2] = image[minY*width + minX].red;
		clusters[i][3] = image[minY*width + minX].green;
		clusters[i][4] = image[minY*width + minX].blue;
		x = x+s;
		if(x>=width){
			y = y+s;
			x = round((double)s/2);
			if(y>=height){
				numSupPix = i+1;
				break;
			}
		}
	}
	// printf("%d Clusters Initialised\n", numSupPix);
	// Identification of Overlap

	// vector<vector<int> >initClosestClust(width,vector<int>(height));
	int **clustMembership, *initClosestClust;

	initClosestClust = (int*)malloc(sizeof(int)*height*width);
	clustMembership = (int**)malloc(sizeof(int*)*height*width);

	int numInitDet = 0, numUnDet = 0, numFuzDet = 0, numRemDet = 0, numPostDet = 0;
	for(j = 0;j<height;j++){
		for(i=0;i<width;i++){
			clustMembership[j*width + i] = (int*)malloc(9*sizeof(int) + 1);
			clustMembership[j*width + i][0] = 0;
			int k, closest;
			double minDist = 1e10;
			for(k = 0;k<numSupPix;k++){
				if(k<9){
					clustMembership[j*width + i][k + 1] = 0;
				}
				if( abs(clusters[k][0]-i) < s && abs(clusters[k][1] - j)<s){
					double dist = pow(pow((clusters[k][0]-i),2)+pow((clusters[k][1] - j),2),0.5);
					if(dist<minDist){
						minDist = dist;
						closest = k;
					}
					clustMembership[j*width + i][clustMembership[j*width + i][0]+1] = k;
					clustMembership[j*width + i][0]++;
				}
			}
			initClosestClust[j*width + i] = closest;

		}
	}
	// printf("Search regions assigned\n");
	// map<int*, int*> overlapRegions; // Overlapping Clusters: Pixels-count, x1,y1,x2,y2
	// for(j = 0;j<height;j++){
	// 	for(i=0;i<width;i++){
	// 		if(clustMembership[j*width + i][0] == 1){
	// 			int k;
	// 			for(k = 0;k<9;k++){
	// 				if(clustMembership[j*width + i][k+1] != 0){
	// 					pixClust[j*width + i] = clustMembership[j*width + i][k+1];
	// 				}
	// 			}
	// 			numInitDet++;
	// 			continue;
	// 		}
	// 		numRemDet++;
	// 		if(overlapRegions.find(clustMembership[j*width + i]) == overlapRegions.end()){
	// 			int* region = (int*)malloc(2*2*s*2*s + 1);
	// 			region[0] = 1;
	// 			region[1] = i;
	// 			region[2] = j;
	// 			overlapRegions.insert( make_pair(clustMembership[j*width + i], region) );
	// 		}
	// 		else{
	// 			overlapRegions[clustMembership[j*width + i]][1 + 2*overlapRegions[clustMembership[j*width + i]][0]] = i;
	// 			overlapRegions[clustMembership[j*width + i]][2 + 2*overlapRegions[clustMembership[j*width + i]][0]] = j;
	// 			overlapRegions[clustMembership[j*width + i]][0]++;
	// 		}
	// 	}
	// }
	// free(clustMembership);
	// // printf("Overlapping Search regions Found\n");
	//
	// map<int*, int*>::iterator itRegion = overlapRegions.begin();
	// // Computation of Degree of Membership and new cluster centers
	// double*** WholeDoM = (double***)malloc(sizeof(double**)*overlapRegions.size()); // For every region, every pixel, dom of every overlap cluster
	// int DoMidx = 0, numMemReq = 0;
	// for(;itRegion!=overlapRegions.end();itRegion++){
	// 	int k,it,l;
	// 	double** DoM = (double**)malloc(sizeof(double*)*itRegion->second[0]); // For every pixel, dom of every overlap cluster
	// 	for(it = 0;it<maxIter;it++){
	// 		//DOM Computation
	// 		for(l =0;l< itRegion->second[0];l++){ //l - pixel
	// 			if(it == 0){
	// 				numMemReq++;
	// 			}
	// 			int x = itRegion->second[2*l+1];
	// 			int y = itRegion->second[2*l+2];
	//
	// 			double* dist = (double*)malloc(sizeof(double)*itRegion->first[0]);
	// 			double* PixelDoM;
	// 			if(it == 0){
	// 				PixelDoM = (double*)malloc(sizeof(double)*itRegion->first[0]);
	// 			}
	// 			else{
	// 				PixelDoM = DoM[l];
	// 			}
	// 			for(k = 0;k<itRegion->first[0];k++){ // k - cluster
	// 				int clustIdx = itRegion->first[k+1];
	// 				double dxy = pow(x-clusters[clustIdx][0],2) + pow(y-clusters[clustIdx][1],2);
	// 				double drgb = pow(image[y*width +x].red - clusters[clustIdx][2],2) + pow(image[y*width +x].blue - clusters[clustIdx][3],2) + pow(image[y*width +x].green - clusters[clustIdx][4],2);
	// 				dist[k] = pow( dxy/pow(mpol,2) + drgb/pow(s,2),0.5);
	// 			}
	// 			double uSum = 0;
	// 			int zeroCount = 0;
	// 			for(k = 0;k<itRegion->first[0];k++){
	// 				if(dist[k] == 0){
	// 					zeroCount++;
	// 				}
	// 				uSum += pow( 1/(dist[k]) ,2/(m-1));
	// 			}
	// 			if(zeroCount == 0){
	// 				for(k =0;k<itRegion->first[0];k++){
	// 					PixelDoM[k] = (1/(pow(dist[k],2/(m-1))*uSum));
	// 				}
	// 			}
	// 			else if(zeroCount == 1){
	// 				for(k =0;k<itRegion->first[0];k++){
	// 					if(dist[k] == 0)
	// 						PixelDoM[k] = (1);
	// 					else{
	// 						PixelDoM[k] = (0);
	// 					}
	// 				}
	// 			}
	// 			else{
	// 				for(k =0;k<itRegion->first[0];k++){
	// 					if(initClosestClust[y*width + x] == itRegion->first[k+1])
	// 						PixelDoM[k] = (1);
	// 					else{
	// 						PixelDoM[k] = (0);
	// 					}
	// 				}
	// 			}
	// 			// free(dist);
	// 			DoM[l] = PixelDoM;
	// 		}
	// 		//Cluster re-computation
	// 		for(k = 0;k< itRegion->first[0];k++){ // k - cluster
	// 			int clustIdx = itRegion->first[k+1];
	// 			double numeratorx = 0, numeratory = 0, numeratorr = 0, numeratorg = 0, numeratorb = 0, denominator = 0;
	// 			for(l = 0;l<itRegion->second[0];l++){ //l - pixel
	// 				int x = itRegion->second[2*l+1];
	// 				int y = itRegion->second[2*l+2];
	// 				numeratorx += pow(DoM[l][k],m)*x;
	// 				numeratory += pow(DoM[l][k],m)*y;
	// 				numeratorr += pow(DoM[l][k],m)*image[y*width + x].red;
	// 				numeratorg += pow(DoM[l][k],m)*image[y*width + x].blue;
	// 				numeratorb += pow(DoM[l][k],m)*image[y*width + x].green;
	// 				denominator += pow(DoM[l][k],m);
	// 			}
	// 			if(denominator == 0)
	// 			 	continue;
	//
	// 			clusters[clustIdx][0] = numeratorx/denominator;
	// 			clusters[clustIdx][1] = numeratory/denominator;
	// 			clusters[clustIdx][2] = numeratorr/denominator;
	// 			clusters[clustIdx][3] = numeratorg/denominator;
	// 			clusters[clustIdx][4] = numeratorb/denominator;
	// 		}
	// 		if(it == maxIter-1){
	// 			WholeDoM[DoMidx++] = DoM;
	// 			break;
	// 		}
	// 	}
	// }
	// free(initClosestClust);
	// // printf("Cluster adjustment and DoM found\n" );
	//
	// double **coords, *Udiff, *forMed;
	// coords = (double**)malloc(sizeof(double*)*numMemReq);
	// Udiff = (double*)malloc(sizeof(double)*numMemReq);
	// forMed = (double*)malloc(sizeof(double)*numMemReq);
	// i = 0;
	// int idx = 0;
	// for(itRegion = overlapRegions.begin();itRegion!=overlapRegions.end();itRegion++){ // Region
	// 	for(j=0;j<itRegion->second[0];j++){ // Pixel
	// 		// Find max and submax WholeDoM[i][j]
	// 		int maxIdx,k;
	// 		double usubmax = 0, umax = 0;
	// 		for(k = 0;k<itRegion->first[0];k++){ // Cluster
	// 			if(WholeDoM[i][j][k]>umax){
	// 				maxIdx = itRegion->first[k+1];
	// 				umax = WholeDoM[i][j][k];
	// 			}
	// 		}
	// 		for(k = 0;k<itRegion->first[0];k++){
	// 			if(WholeDoM[i][j][k]>usubmax && itRegion->first[k+1] != maxIdx){
	// 				usubmax = WholeDoM[i][j][k];
	// 			}
	// 		}
	// 		free(WholeDoM[i][j]);
	// 		double* params = (double*)malloc(3*sizeof(double));
	// 		params[0] = itRegion->second[2*j+1];
	// 		params[1] = itRegion->second[2*j+2];
	// 		params[2] = maxIdx;
	// 		coords[idx] = params;
	// 		Udiff[idx++] = umax-usubmax;
	//
	// 	}
	// 	free(itRegion->first);
	// 	free(itRegion->second);
	// 	free(WholeDoM[i]);
	// 	i++;
	// }
	// free(WholeDoM);
	// // printf("DoMs computed and compiled\n" );
	// memcpy(forMed, Udiff, sizeof(double)*numMemReq);
	// sort(forMed,forMed+numMemReq);
	// double median = forMed[(int)numMemReq/2];
	// free(forMed);
	// // cout<<"Median: "<<median<<endl;
	// for(i = 0;i<numMemReq;i++){
	// 	int x = coords[i][0];
	// 	int y = coords[i][1];
	// 	int clust = coords[i][2];
	// 	if(Udiff[i]<median){
	// 		numUnDet ++;
	// 		int numSearchSup = 0;
	// 		int ClustIdx;
	// 		for(j = 0;j<numSupPix;j++){
	// 			if(abs(clusters[j][0]-x) < M/2 && abs(clusters[j][1]-y)<M/2){
	// 				numSearchSup++;
	// 				ClustIdx = j;
	// 				numPostDet++;
	// 			}
	// 		}
	//
	// 		if(numSearchSup == 1){
	// 			pixClust[y*width + x] = ClustIdx;
	// 			numPostDet++;
	// 		}
	// 		else{
	// 			pixClust[y*width + x] = -1;;
	// 		}
	//
	// 	}
	// 	else{
	// 		numFuzDet++;
	// 		pixClust[y*width + x] = clust;
	//
	// 	}
	// 	free(coords[i]);
	// }
	// free(coords);
	// free(Udiff);
	// // printf("Pixels assigned\n" );
	//
	// // printf("Total: %d, Initialised Det: %d, Rem Det: %d, Fuzzy Det: %d, PostDet: %d, Undet: %d, Median %lf\n",512*512,numInitDet, numRemDet,numFuzDet,numPostDet, numUnDet, median);
	//
	// double** clustMeans = (double**)malloc(sizeof(double*)*numSupPix);
	// for(i = 0;i<numSupPix;i++){
	// 	clustMeans[i] = (double*)malloc(sizeof(double)*4);
	// 	clustMeans[i][0] = 0;
	// }
	// for(i= 0;i<width*height;i++){
	// 	if(pixClust[i] != -1){
	// 		clustMeans[pixClust[i]][0]++;
	// 	}
	// }
	// for(i= 0;i<width;i++){
	// 	for(j = 0;j<height;j++){
	// 		if(pixClust[j*width + i] != -1){
	// 			clustMeans[pixClust[j*width + i]][1] += image[j*width + i].red/clustMeans[pixClust[j*width + i]][0];
	// 			clustMeans[pixClust[j*width + i]][2] += image[j*width + i].green/clustMeans[pixClust[j*width + i]][0];
	// 			clustMeans[pixClust[j*width + i]][3] += image[j*width + i].blue/clustMeans[pixClust[j*width + i]][0];
	// 		}
	// 	}
	// }
	// int count = 0;
	// for(i = 0;i<numSupPix;i++){
	// 	count += (clustMeans[i][0]==0);
	//
	// }
	// *clustMeansPtr = clustMeans;
}

int main()
{
	int maxIter = 10;
	double mpol = 0.1, m = 3,M = 5; // CONSTANTS

	char inputFile[30];
	PPMImage* image;
	int numSupPix = 900;
	int maxSize = 2048, size;
	int total = maxSize;

	for(size = 32;size<=maxSize;size*=2){
		int RUNS = maxSize/size,i;
		sprintf(inputFile,"../Data/%dx%d.ppm\0",size,size);
		image = readPPM(inputFile);

		double** clustMeans;// Stores in the information regarding belonging to a cluster - needs changes; Clusters -  0 - x, 1 - y, 2 - r, 3 - g, 4 - 5 Stores the information about the cluster centers
		int *pixClust; // For Pixel memberships

		pixClust = (int*)malloc(sizeof(int)*image->x*image->y);
		int s = ceil(sqrt(size*size/numSupPix));
		// cout<<"Superpixel Size: "<<s<<endl;
		// printf("Algorithm Starting\n");
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		for(i = 0;i<RUNS*RUNS;i++){
			FuzzyS(image->data, image->x, image->y, &clustMeans, pixClust, s, numSupPix, maxIter, mpol, m,M);
		}
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);

		float tot_time_ms;
		cudaEventElapsedTime(&tot_time_ms,start,stop);
		double tot_time = ((double)tot_time_ms/1000);

		double throughput = (sizeof(double)*total*total)/tot_time; // NEED TO EDIT
		printf("Size: %lld, Time: %f, Throughput: %f\n",size,tot_time,throughput);

		free(clustMeans);
		free(pixClust);
	}
}
