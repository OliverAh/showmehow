#!/usr/bin/env nextflow

workflow {
    GENERATE_PARAMS(file("$moduleDir/process0/create_params.py"))
    MATMUL_CUPY1(file("$moduleDir/process1/scriptfile.py"), GENERATE_PARAMS.out.param_files.flatten())
    MATMUL_CUPY2(file("$moduleDir/process2/scriptfile.py"), GENERATE_PARAMS.out.param_files.flatten())
}


process GENERATE_PARAMS {
    conda "$moduleDir/process0/environment.yml"

    input:
        path scriptfile

    output:
        path "params_*.txt", emit: param_files
    
    script:
    """
    python ${scriptfile}
    """
}

process MATMUL_CUPY1 {
    conda "$moduleDir/process1/environment.yml"
    maxForks 1

    input:
        path script_file
        path params_file

    // output:
    //     path "output_*.txt"

    script:
    """
    python ${script_file} ${params_file}
    """
}

process MATMUL_CUPY2 {
    conda "$moduleDir/process2/environment.yml"
    maxForks 1
    
    input:
        path script_file
        path params_file

    // output:
    //     path "output_*.txt"

    script:
    """
    python ${script_file} ${params_file}
    """
}





