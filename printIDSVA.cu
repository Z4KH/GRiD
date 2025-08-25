// nvcc -gencode arch=compute_89,code=sm_89 -o printIDSVA printIDSVA.cu


#include <random>
#include <algorithm>
#include "grid.cuh"
#define RANDOM_MEAN 0
#define RANDOM_STDEV 1
std::default_random_engine randEng(1337); // fixed seed
std::normal_distribution<double> randDist(RANDOM_MEAN, RANDOM_STDEV); //mean followed by stdiv
template <typename T>
T getRand(){return static_cast<T>(randDist(randEng));}








template <typename T>
__host__
void test(){
	T gravity = static_cast<T>(9.81);
	dim3 dimms(grid::SUGGESTED_THREADS,1,1);
	
	cudaStream_t *streams = grid::init_grid<T>();
	
	grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
	
	grid::gridData<T> *hd_data = grid::init_gridData<T,1>();
	
	
	
	T q[] = {0.300623, -1.427442, 0.047334, -0.51204, -1.437442, 0.500384, -0.881586};
	T qd[] = {-1.226503, -0.619695, 0.973148, -0.750689, -0.253769, 0.493305, -0.695605};
	T qdd[] = {0.425334, 0.340006, -0.178834, -0.013169, -2.349815, 0.405039, -2.266609};
	// T q[] = {0.300623, -1.427442, 0.047334, -0.51204, -1.437442, 0.500384, -0.881586, -1.226503, -0.619695, 0.973148, -0.750689, -0.253769};
	// T qd[] = {0.493305, -0.695605, 0.425334, 0.340006, -0.178834, -0.013169, -2.349815, 0.405039, -2.266609, -0.424634, 1.034167, -0.270165};
	// T qdd[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	
	// load q,qd,u
	for(int j = 0; j < grid::NUM_JOINTS; j++){
		// hd_data->h_q_qd_u[j] = getRand<double>(); 
		// hd_data->h_q_qd_u[j+grid::NUM_JOINTS] = getRand<double>(); 
		// hd_data->h_q_qd_u[j+2*grid::NUM_JOINTS] = getRand<double>();
		hd_data->h_q_qd_u[j] = q[j];; 
		hd_data->h_q_qd_u[j+grid::NUM_JOINTS] = qd[j];; 
		hd_data->h_q_qd_u[j+2*grid::NUM_JOINTS] = qdd[j];
	}
	gpuErrchk(cudaMemcpy(hd_data->d_q_qd_u,hd_data->h_q_qd_u,3*grid::NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());

	grid::inverse_dynamics_single_timing<T>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
	grid::inverse_dynamics_single_timing<T>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
	grid::inverse_dynamics_gradient_single_timing<T>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
	grid::idsva_so_host_single_timing<T>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
	
	// // Print Results
	printf("d2tau_dq2\n");
	for (int i = 0; i < grid::NUM_JOINTS; i++){
		printf("Joint %i\n",i);
		printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(&hd_data->h_idsva_so[i*grid::NUM_JOINTS*grid::NUM_JOINTS],grid::NUM_JOINTS);
		printf("\n\n");
	}
	printf("\n\n\n\nd2tau_dqd2\n");
	for (int i = 0; i < grid::NUM_JOINTS; i++){
		printf("Joint %i\n",i);
		printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(&hd_data->h_idsva_so[grid::NUM_JOINTS*grid::NUM_JOINTS*grid::NUM_JOINTS+i*grid::NUM_JOINTS*grid::NUM_JOINTS],grid::NUM_JOINTS);
		printf("\n\n");
	}
	printf("\n\n\n\nd2tau_cross\n");
	for (int i = 0; i < grid::NUM_JOINTS; i++){
		printf("Joint %i\n",i);
		printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(&hd_data->h_idsva_so[2*grid::NUM_JOINTS*grid::NUM_JOINTS*grid::NUM_JOINTS+i*grid::NUM_JOINTS*grid::NUM_JOINTS],grid::NUM_JOINTS);
		printf("\n\n");
	}
	printf("\n\n\n\ndM_dq\n");
	for (int i = 0; i < grid::NUM_JOINTS; i++){
		printf("Joint %i\n",i);
		printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(&hd_data->h_idsva_so[3*grid::NUM_JOINTS*grid::NUM_JOINTS*grid::NUM_JOINTS+i*grid::NUM_JOINTS*grid::NUM_JOINTS],grid::NUM_JOINTS);
		printf("\n\n");
	}


	grid::close_grid<T>(streams,d_robotModel,hd_data);
}

int main(void){
	test<float>(); 
	return 0;
}
