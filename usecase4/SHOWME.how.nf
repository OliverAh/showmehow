#!/usr/bin/env nextflow


process PrePost {

    conda "$moduleDir/01_PrePost/mamba_env.yml"
    // publishDir "$moduleDir/outputs", mode: 'copy'

    input:
      path script

    output:
      path "input_files/*", emit: input_files
      val true, emit:rdy

    script:
    """
    python ${script}
    """

}

process InstantiateGPUs {

    input:
      val gpu_ids
      val someVal
    
    script:
    // mkdir -p $workDir/GPUsToUse
    """
    rm -fv $workDir/GPUsToUse/* || true
    gpu_ids=`echo $gpu_ids | tr -d [ | tr -d ] | tr -d ,`
    echo "GPU IDs: \$gpu_ids"
    sleep 5
    for gpu_id in \$gpu_ids; do
    echo "GPU ID: \$gpu_id"
    touch $workDir/GPUsToUse/\$gpu_id
    done
    """
  
}

process SimQiskit {
    conda "$moduleDir/02_Qiskit/mamba_env_qiskit.yml"

    input:
      path script
      path params_file
      val gpu_id

    output:
      path "counts_array_id_*.npz", emit: sim_out
      // val GPUID, emit: gpu_id

    script:
    """
    rm -f $workDir/GPUsToUse/${gpu_id.baseName}
    CUDA_VISIBLE_DEVICES=${gpu_id.baseName} python ${script}
    touch $workDir/GPUsToUse/${gpu_id.baseName}
    """
    
    }

process SimPennylane {
    conda "$moduleDir/02_Pennylane/mamba_env_pennylane.yml"

    input:
      path script
      path qiskit_script
      path params_file
      val gpu_id
      val dontneed

    output:
      path "counts_array_id_*.npz", emit: sim_out
      // val GPUID, emit: gpu_id

    script:
    """
    rm -f $workDir/GPUsToUse/${gpu_id.baseName}
    CUDA_VISIBLE_DEVICES=${gpu_id.baseName} python ${script}
    touch $workDir/GPUsToUse/${gpu_id.baseName}
    """
    
    }

process Finalize {
  input:
    // val somevalthatweactuallydontneed
    val somemorevalthatweactuallydontneed

  script:
  """
  touch $workDir/GPUsToUse/finished
  """
}

// process GpuScheduler{

//   ????????????????

// }

// process CollectResults {

//   publishDir "$moduleDir/outputs"
//   ???????????
  

// }


workflow {

    def GPUsToUseDir = new File("${workDir}/GPUsToUse")
    if(!GPUsToUseDir.exists()) {
        GPUsToUseDir.mkdir()
    }

    PrePost(
      file ("$moduleDir/01_PrePost/01_HHL_param_prep.py")
    )

    AvailGPUs = channel.watchPath("${workDir}/GPUsToUse/*", 'create')
    // .until{v -> v.baseName=="finished"}

    // AvailSims = AvailGPUs.map{v -> ['q','p']}
    AvailSims = AvailGPUs.branch { v ->
        q: SimPennylane.out.collect()
        p: SimQiskit.out.collect()
        other: true
        }
        .set{AvailSims2}

    AvailSims2.q.view{"ended up in q"}
    AvailSims2.p.view{"ended up in p"}
    AvailSims2.other.view{"ended up in other"}

    // AvailMerged = AvailGPUs.merge(AvailSims.flatten()).view()


    GPUsToUse = channel.of([0,1,2,3])
    
    InstantiateGPUs(
      GPUsToUse,
      channel.of(1).concat(AvailGPUs).flatten()
    )


    SimQiskit(
      file ("$moduleDir/02_Qiskit/02_HHL_Qiskit_nextflow.py"),
      PrePost.out.input_files.flatten(),
      AvailGPUs.flatten()
      )
    
    AvailGPUs2 = SimQiskit.out.collect().filter(false).concat(channel.watchPath("${workDir}/GPUsToUse/*", 'create'))

    SimPennylane(
      file ("$moduleDir/02_Pennylane/HHL_Pennylane_nextflow.py"),
      file ("$moduleDir/02_Qiskit/HHL_Qiskit_nextflow.py"),
      PrePost.out.input_files.flatten(),
      AvailGPUs2.flatten(),
      SimQiskit.out.collect()
      )

    Finalize(
      SimPennylane.out.collect()
    )
    //   AvailGPUs.concat(SimQiskit.out.gpu_id)


}