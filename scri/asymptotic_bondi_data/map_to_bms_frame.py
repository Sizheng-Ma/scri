import os
import scri
import numpy as np
import quaternion
import spherical_functions as sf

from spherical_functions import LM_index as lm

from scipy.optimize import minimize

import ast
import json

from quaternion.calculus import indefinite_integral as integrate

def MT_to_WM(h_mts, dataType=scri.h):
    h = scri.WaveformModes(t=h_mts.t,\
                           data=np.array(h_mts)[:,lm(h_mts.ell_min, -h_mts.ell_min, h_mts.ell_min):lm(h_mts.ell_max+1, -(h_mts.ell_max+1), h_mts.ell_min)],\
                           ell_min=h_mts.ell_min,\
                           ell_max=h_mts.ell_max,\
                           frameType=scri.Inertial,\
                           dataType=dataType
                          )
    return h

def time_after_half_orbits(h, number_of_half_orbits, start_time = None):
    a = np.angle(scri.to_coprecessing_frame(h.copy()).data[:,lm(2,2,h.ell_min)])
    maxs = np.array(np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True], dtype=bool)
    maxs_after = maxs[np.argmin(abs(h.t-start_time)):]
    Dt = h.t[np.where(maxs_after[1:] == 1)[0][number_of_half_orbits - 1]] - h.t[np.where(maxs_after[1:] == 1)[0][0]]
    return h.t[np.argmin(abs(h.t-(start_time+Dt)))]

def transformation_from_charge(Q, t, t1, t2):
    idx1 = np.argmin(abs(t - t1))
    idx2 = np.argmin(abs(t - t2))
    space_translation = []
    boost_velocity = []
    for i in range(3):
        polynomial_fit = np.polyfit(t[idx1:idx2], Q[idx1:idx2, i], deg=1)
        space_translation.append(polynomial_fit[1])
        boost_velocity.append(polynomial_fit[1])

    transformation = {
        "space_translation": np.array(space_translation),
        "boost_velocity": np.array(boost_velocity)
    }
    
    return transformation

def transformation_to_map_to_com_frame(abd, t1=None, t2=None, n_iterations=5, padding_time=100, interpolate=False, return_convergence=False):
    if t1 == None:
        t1 = time_after_half_orbits(MT_to_WM(2.0*abd.sigma.bar,scri.h), 6, 200)
    if t2 == None:
        t2 = time_after_half_orbits(MT_to_WM(2.0*abd.sigma.bar,scri.h), 8, t1)

    # interpolate to make things faster
    if interpolate:
        abd = abd.interpolate(abd.t[np.argmin(abs(abd.t - (t1 - padding_time))):np.argmin(abs(abd.t - (t2 + padding_time)))])
    
    G = abd.bondi_comoving_CoM_charge()/abd.bondi_four_momentum()[:, 0, None]
    
    transformation_to_map_to_com = transformation_from_charge(G, abd.t, t1, t2)
    
    transformations_to_map_to_com = [transformation_to_map_to_com]

    for itr in range(n_iterations - 1):
        abd_prime = abd.transform(space_translation = transformation_to_map_to_com["space_translation"],\
                                  boost_velocity = transformation_to_map_to_com["boost_velocity"])

        G_prime = abd_prime.bondi_comoving_CoM_charge()/abd_prime.bondi_four_momentum()[:, 0, None]

        transformation_to_map_to_com_prime = transformation_from_charge(G_prime, abd_prime.t, t1, t2)
        
        for transformation in transformation_to_map_to_com_prime:
            transformation_to_map_to_com[transformation] += transformation_to_map_to_com_prime[transformation]

        transformations_to_map_to_com.append(transformation_to_map_to_com)

    if return_convergence:
        def norm(v):
            return np.sqrt(sum(n * n for n in v))
        
        convergence = {
            "space_translation": np.array([norm(transformations_to_map_to_com[0]["space_translation"])] + \
                                          [0.0]*(len(transformations_to_map_to_com)-1)),
            "boost_velocity": np.array([transformations_to_map_to_com[0]["boost_velocity"]] +\
                                       [0.0]*(len(transformations_to_map_to_com)-1))
        }
        for i in range(1,len(transformations_to_map_to_com)):
            for transformation in convergence:
                convergence[transformation][i] =\
                    norm(transformations_to_map_to_com[i][transformation]) -\
                    norm(transformations_to_map_to_com[i-1][transformation])
        return transformation_to_map_to_com, convergence
    else:
        return transformation_to_map_to_com
    
def supertranslation_initial_guess(abd, h_target, t1, t2, ell):
    averages = []
    for h in [MT_to_WM(2.0*abd.sigma.bar, scri.h), h_target]:
        average_per_mode = []
        for M in range(-ell, ell + 1):
            average_per_mode.append(np.mean(h.data[np.argmin(abs(abd.t - t1)):np.argmin(abs(abd.t - t2)), lm(ell, M, abd.sigma.ell_min)]))
        averages.append(average_per_mode)
        
    # this isn't perfect due to time interpolation, but should be reasonable as an initial guess
    conversion_factor = -0.5
    return (conversion_factor*(np.array(averages)[1,:] - np.array(averages)[0,:])).tolist()

def combine_transformations_to_supertranslation(transformations):
    combined_transformations = {}
    
    supertranslation_modes = []
    for transformation in transformations:
        if transformation == "time_translation":
            supertranslation_modes.append(0)
        if transformation == "space_translation":
            supertranslation_modes.append(1)
        if "supertranslation" in transformation:
            ell = int(transformation.split('ell_')[1])
            supertranslation_modes.append(ell)
            
    if len(supertranslation_modes) > 0:
        supertranslation = np.zeros(int((max(supertranslation_modes) + 1)**2.0)).tolist()
        for transformation in transformations:
            if transformation == "time_translation":
                supertranslation[0] = sf.constant_as_ell_0_mode(transformations[transformation][0])
            if transformation == "space_translation":
                supertranslation[1:4] = -sf.vector_as_ell_1_modes(np.array(transformations[transformation]))
            if transformation == "frame_rotation":
                combined_transformations[transformation] = transformations[transformation]
            if transformation == "boost_velocity":
                combined_transformations[transformation] = transformations[transformation]
            if "supertranslation" in transformation:
                ell = int(transformation.split('ell_')[1])
                supertranslation[int(ell**2):int(ell**2 + (2*ell+1))] = transformations[transformation]
            
        combined_transformations["supertranslation"] = supertranslation

    return combined_transformations

def transform_abd(abd, transformations):
    if "supertranslation" in transformations:
        if "frame_rotation" in transformations:
            if "boost_velocity" in transformations:
                abd_prime = abd.transform(supertranslation=transformations["supertranslation"],\
                                          frame_rotation=transformations["frame_rotation"],\
                                          boost_velocity=transformations["boost_velocity"])
            else:
                abd_prime = abd.transform(supertranslation=transformations["supertranslation"],\
                                          frame_rotation=transformations["frame_rotation"])
        elif "boost_velocity" in transformations:
            abd_prime = abd.transform(supertranslation=transformations["supertranslation"],\
                                      boost_velocity=transformations["boost_velocity"])
        else:
            abd_prime = abd.transform(supertranslation=transformations["supertranslation"])
    elif "frame_rotation" in transformations:
        if "boost_velocity" in transformations:
            abd_prime = abd.transform(frame_rotation=transformations["frame_rotation"],\
                                      boost_velocity=transformations["boost_velocity"])
        else:
            abd_prime = abd.transform(frame_rotation=transformations["frame_rotation"])
    elif "boost_velocity" in transformations:
        abd_prime = abd.transform(boost_velocity=transformations["boost_velocity"])
    else:
        abd_prime = abd.copy()
    return abd_prime

def obtain_previous_map_to_bms(transformations, json_file):
    previous_map_to_bms = {}
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            map_to_bms_from_file = json.load(f)["transformations"]
        for transformation in transformations:
            if transformation in map_to_bms_from_file:
                previous_map_to_bms[transformation] = ast.literal_eval(map_to_bms_from_file[transformation])
    return previous_map_to_bms

def obtain_map_to_bms_initial_guess(previous_map_to_bms, new_transformation, abd, h_target, t1, t2, use_educated_guess=True):
    initial_guess = previous_map_to_bms.copy()

    combined_transformations = combine_transformations_to_supertranslation(previous_map_to_bms)
    
    abd_prime = transform_abd(abd, combined_transformations)

    if new_transformation == "time_translation":
        initial_guess[new_transformation] = [0.0]
    if new_transformation == "space_translation":
        if use_educated_guess:
            transformation_to_map_to_com = transformation_to_map_to_com_frame(abd_prime, t1, t2)
            initial_guess[new_transformation] = transformation_to_map_to_com[new_transformation]
        else:
            initial_guess[new_transformation] = [0.0]*3
    if new_transformation == "frame_rotation":
        initial_guess[new_transformation] = [1.0, 0.0, 0.0, 0.0]
    if new_transformation == "boost_velocity":
        if use_educated_guess:
            transformation_to_map_to_com = transformation_to_map_to_com_frame(abd_prime, t1, t2)
            initial_guess[new_transformation] = transformation_to_map_to_com[new_transformation]
        else:
            initial_guess[new_transformation] = [0.0]*3
    if "supertranslation" in new_transformation:
        ell = int(new_transformation.split('ell_')[1])
        if use_educated_guess:
            initial_guess[new_transformation] = supertranslation_initial_guess(abd_prime, h_target, t1, t2, ell)
        else:
            initial_guess[new_transformation] = [0.0]*int(2.0*ell + 1)
        
    return initial_guess

def as_solver_input(map_to_bms_initial_guess, transformations, bounds):
    def as_reals_and_complexes(x, ell_min, ell_max):
        reverted_input = []
        for L in range(ell_min, ell_max+1):
            for M in range(-L,0+1):
                reverted_input.append(x[lm(L,M,ell_min)].real)
        for L in range(ell_min, ell_max+1):
            for M in range(-L,0):
                reverted_input.append(x[lm(L,M,ell_min)].imag)
        return reverted_input
    
    solver_input = []
    solver_bounds = []
    solver_constraint_keys = []
    frame_rotation_idx = None
    boost_velocity_idx = None
    for transformation in transformations:
        if transformation == "time_translation":
            solver_input += map_to_bms_initial_guess[transformation]
        if transformation == "space_translation":
            solver_input += np.array(map_to_bms_initial_guess[transformation]).tolist()
        if transformation == "frame_rotation":
            # we only need the final three components
            solver_input += map_to_bms_initial_guess[transformation][1:4]
            frame_rotation_idx = len(solver_input) - 3
            solver_constraint_keys.append({'type': 'ineq', 'fun': lambda x: 1.0 - sum(n * n for n in x[frame_rotation_idx:frame_rotation_idx + 3])})
        if transformation == "boost_velocity":
            solver_input += np.array(map_to_bms_initial_guess[transformation]).tolist()
            boost_velocity_idx = len(solver_input) - 3
            solver_constraint_keys.append({'type': 'ineq', 'fun': lambda x: 1.0 - sum(n * n for n in x[boost_velocity_idx:boost_velocity_idx + 3])})
        if "supertranslation" in transformation:
            ell = int(transformation.split('ell_')[1])
            solver_input += as_reals_and_complexes(map_to_bms_initial_guess[transformation], ell, ell)
            
        for bound in bounds[transformation]:
            solver_bounds.append(bound)

    return solver_input, solver_bounds, tuple(solver_constraint_keys), frame_rotation_idx, boost_velocity_idx

def as_map_to_bms(x0, transformations, to_constant_and_vector=True):
    def to_modes(x, ell_min, ell_max):
        def fix_idx(L,ell_min):
            return int((L-ell_min)*(L+ell_min-1)/2)
        
        fixed_input = np.zeros((ell_max + 1)**2 - (ell_min)**2, dtype=complex)
        for L in range(ell_min, ell_max+1):
            for M in range(-L,0+1):
                if M == 0:
                    fixed_input[lm(L,M,ell_min)] = x[lm(L,M,ell_min)-fix_idx(L,ell_min)]
                else:
                    fixed_input[lm(L,M,ell_min)] = x[lm(L,M,ell_min)-fix_idx(L,ell_min)] +\
                        1.0j*x[(lm(ell_max,ell_max,ell_min)-fix_idx(ell_max+1,ell_min)+1)+\
                                lm(L,M,ell_min)-fix_idx(L,ell_min)-L+ell_min]
            for M in range(1,L+1):
                fixed_input[lm(L,M,ell_min)] = (-1)**M * np.conj(fixed_input[lm(L,-M,ell_min)])
        return fixed_input
    

    map_to_bms = {}

    idx_itr = 0
    supertranslation_modes = []
    for transformation in transformations:
        if transformation == "time_translation":
            supertranslation_modes.append(0)
        if transformation == "space_translation":
            supertranslation_modes.append(1)
        if "supertranslation" in transformation:
            ell = int(transformation.split('ell_')[1])
            supertranslation_modes.append(ell)
            
    supertranslation = np.zeros(int((max(supertranslation_modes) + 1)**2.0)).tolist()
    for transformation in transformations:
        if transformation == "time_translation":
            if to_constant_and_vector:
                map_to_bms[transformation] = x0[idx_itr:idx_itr + 1]
            else:
                supertranslation[0] = sf.constant_as_ell_0_mode(x0[idx_itr:idx_itr + 1])
            idx_itr += 1
        if transformation == "space_translation":
            if to_constant_and_vector:
                map_to_bms[transformation] = x0[idx_itr:idx_itr + 3]
            else:
                supertranslation[1:4] = -sf.vector_as_ell_1_modes(x0[idx_itr:idx_itr + 3])
            idx_itr += 3
        if transformation == "frame_rotation":
            map_to_bms[transformation] = [np.sqrt(1.0 - sum(n * n for n in x0[idx_itr:idx_itr + 3]))] + [n for n in x0[idx_itr:idx_itr + 3]]
            idx_itr += 3
        if transformation == "boost_velocity":
            map_to_bms[transformation] = x0[idx_itr:idx_itr + 3]
            idx_itr += 3
        if "supertranslation" in transformation:
            ell = int(transformation.split('ell_')[1])
            if to_constant_and_vector:
                map_to_bms[transformation] = to_modes(x0[idx_itr:idx_itr + int(2.0*ell + 1)], ell, ell)
            else:
                supertranslation[int(ell**2):int(ell**2 + (2*ell+1))] = to_modes(x0[idx_itr:idx_itr + int(2.0*ell + 1)], ell, ell)
            idx_itr += int(2.0*ell + 1)
            
    if not to_constant_and_vector:
        map_to_bms["supertranslation"] = supertranslation
    
    return map_to_bms

def L2_norm(abd_prime, h_target, t1, t2):
    h_prime = MT_to_WM(2.0*abd_prime.sigma.bar, scri.h).interpolate(\
                        abd_prime.t[np.argmin(abs(abd_prime.t - t1)):np.argmin(abs(abd_prime.t - t2))])

    h_target_prime = h_target.interpolate(h_prime.t)

    ell_max = min(h_prime.ell_max, h_target_prime.ell_max)

    diff = h_prime.copy()
    diff.data = diff.data[:,lm(2,-2,diff.ell_min):lm(ell_max + 1, -(ell_max + 1), diff.ell_min)] -\
        h_target_prime.data[:,lm(2,-2,h_target_prime.ell_min):lm(ell_max + 1, -(ell_max + 1), h_target_prime.ell_min)]
    diff.ell_min = 2
    diff.ell_max = ell_max

    return integrate(diff.norm(), diff.t)[-1]

def minimize_L2_norm(x0, abd, h_target, t1, t2, transformations, frame_rotation_idx=None, boost_velocity_idx=None):
    if x0 != []:
        if frame_rotation_idx != None:
            if 1.0 - sum(n * n for n in x0[frame_rotation_idx:frame_rotation_idx + 3]) < 0:
                return 1e6
        if boost_velocity_idx != None:
            if 1.0 - sum(n * n for n in x0[boost_velocity_idx:boost_velocity_idx + 3]) < 0:
                return 1e6
            
        map_to_bms = as_map_to_bms(x0, transformations, to_constant_and_vector=False)

        abd_prime = transform_abd(abd, map_to_bms)
    else:
        abd_prime = abd.copy()

    return L2_norm(abd_prime, h_target, t1, t2)

def write_map_to_bms(output_map_to_bms, times, errors, json_file):
    default_order = ["time_translation","space_translation","frame_rotation","boost_velocity"]
    
    reordered_map_to_bms = {
        "times": {
        },
        "transformations": {
        },
        "errors": {
        },
    }

    # times
    for time in times:
        reordered_map_to_bms["times"][time] = str(np.array(times[time]).tolist())

    # transformations
    for transformation in default_order:
        if transformation in output_map_to_bms:
            reordered_map_to_bms["transformations"][transformation] = str(np.array(output_map_to_bms[transformation]).tolist())

    supertranslations = []
    for transformation in output_map_to_bms:
        if "supertranslation" in transformation:
            supertranslations.append(transformation)
    supertranslations = np.sort(supertranslations)
    for transformation in supertranslations:
        reordered_map_to_bms["transformations"][transformation] = str(np.array(output_map_to_bms[transformation]).tolist())

    # errors
    for error in errors:
        reordered_map_to_bms["errors"][error] = str(np.array(errors[error]).tolist())

    with open(json_file, 'w') as f:
        json.dump(reordered_map_to_bms, f, indent=2, separators=(",", ": "), ensure_ascii=True)

def map_to_bms_frame(self, h_target, json_file, map_to_bms=None, t1=None, t2=None, bounds=None):
    """Map an AsymptoticBondiData object to the BMS frame of another object

    Parameters
    ==========
    abd: AsymptoticBondiData
        The object storing the modes of the original data, which will be transformed in this
        function.  This is the only required argument to this function.
    transformations_to_use: string array, optional
        Defaults to ['Poincare', 'supertranslation_ell_2_4']. 
    t1: float, optional
        Defaults to three orbits past t=200M.
    t2: float, optional
        Defaults to four orbits past t1.

    Returns
    -------
    abdprime: AsymptoticBondiData
        Object representing the transformed data.

    """

    # transformations to optimize
    if map_to_bms == None:
        map_to_bms = {
            "supertranslation_ell_2": None,
            "time_translation": None,
            "space_translation": None,
            "frame_rotation": None,
            "boost_velocity": None,
            "supertranslation_ell_3": None,
            "supertranslation_ell_4": None
        }

    if t1 == None:
        t1 = time_after_half_orbits(MT_to_WM(2.0*self.sigma.bar,scri.h), 6, 200)
    if t2 == None:
        t2 = time_after_half_orbits(MT_to_WM(2.0*self.sigma.bar,scri.h), 8, t1)
        
    if bounds == None:
        bounds = map_to_bms.copy()
        default_bounds = {
            "time_translation": [(-100.0, 100.0)],
            "space_translation": [(-1e-2, 1e-2)]*3,
            "frame_rotation": [(-1.0, 1.0)]*3,
            "boost_velocity": [(-1e-4, 1e-4)]*3,
            "supertranslation_ell_2": [(-1.0, 1.0)]*5,
            "supertranslation_ell_3": [(-1.0, 1.0)]*7,
            "supertranslation_ell_4": [(-1.0, 1.0)]*9
        }
        for transformation in bounds:
            bounds[transformation] = default_bounds[transformation]

    # interpolate to make things faster
    padding_time = 100
    abd = self.interpolate(self.t[np.argmin(abs(self.t - (t1 - padding_time))):np.argmin(abs(self.t - (t2 + padding_time)))])
    
    error1 = minimize_L2_norm([], abd, h_target, t1, t2, list(map_to_bms.keys()))

    print("")

    print("Initial Error: ", error1, "\n")

    use_previous_maps = True
    if use_previous_maps:
        start_idx = len(list(obtain_previous_map_to_bms(list(map_to_bms.keys()), json_file).keys()))
    else:
        start_idx = 0
    for idx in range(start_idx, len(list(map_to_bms.keys()))):
        transformations = list(map_to_bms.keys())[:(idx + 1)]

        print("Transformations: ", transformations, "\n")

        previous_map_to_bms = obtain_previous_map_to_bms(transformations[:idx], json_file)

        print("Previous Map: ", previous_map_to_bms, "\n")
        
        map_to_bms_initial_guess = obtain_map_to_bms_initial_guess(previous_map_to_bms, transformations[-1], abd, h_target, t1, t2,\
                                                                   use_educated_guess=True)

        print("IG: ", map_to_bms_initial_guess, "\n")

        solver_initial_guess, solver_bounds, solver_constraints, frame_rotation_idx, boost_velocity_idx = as_solver_input(map_to_bms_initial_guess, transformations, bounds)

        print("IG (solver): ", solver_initial_guess, "\n")

        # remove the other functions from abd
        abd_prime = abd.copy()
        abd_prime.psi0 = 0.0*abd_prime.psi0;
        abd_prime.psi1 = 0.0*abd_prime.psi1;
        abd_prime.psi2 = 0.0*abd_prime.psi2;
        abd_prime.psi3 = 0.0*abd_prime.psi3;
        abd_prime.psi4 = 0.0*abd_prime.psi4;
        
        res = minimize(minimize_L2_norm, x0=solver_initial_guess,
                       args=(abd_prime, h_target, t1, t2, transformations, frame_rotation_idx, boost_velocity_idx),
                       method='SLSQP', bounds=solver_bounds, constraints=solver_constraints,\
                       options={'ftol': 1e-10, 'disp': True})
        
        error2 = minimize_L2_norm(res.x, abd_prime, h_target, t1, t2, transformations, frame_rotation_idx, boost_velocity_idx)
        
        print("Final Error: ", error2, "\n")

        times = {
            "t1": t1,
            "t2": t2
        }
        errors = {
            "error1": error1,
            "error2": error2
        }
        
        write_map_to_bms(as_map_to_bms(res.x, transformations), times, errors, json_file)
    
    return as_map_to_bms(res.x, transformations)
