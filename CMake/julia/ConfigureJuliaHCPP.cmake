# ConfigureJuliaHCPP.cmake: generate an mlpack .h file for a Julia binding given
# input arguments.
#
# This file depends on the following variables being set:
#
#  * PROGRAM_NAME: name of the binding
#  * PROGRAM_MAIN_FILE: the file containing the mlpackMain() function.
#  * JULIA_H_IN: path of the julia_method.h.in file.
#  * JULIA_H_OUT: name of the output .h file.
#  * JULIA_CPP_IN: path of the julia_method.cpp.in file.
#  * JULIA_CPP_OUT: name of the output .cpp file.
#
# We need to parse the main file and find any PARAM_MODEL_* lines.
file(READ "${PROGRAM_MAIN_FILE}" MAIN_FILE)

# Grab all "PARAM_MODEL_IN(Model,", "PARAM_MODEL_IN_REQ(Model,",
# "PARAM_MODEL_OUT(Model,".
string(REGEX MATCHALL "PARAM_MODEL_IN\\([A-Za-z_]*," MODELS_IN "${MAIN_FILE}")
string(REGEX MATCHALL "PARAM_MODEL_IN_REQ\\([A-Za-z_]*," MODELS_IN_REQ
    "${MAIN_FILE}")
string(REGEX MATCHALL "PARAM_MODEL_OUT\\([A-Za-z_]*," MODELS_OUT "${MAIN_FILE}")

string(REGEX REPLACE "PARAM_MODEL_IN\\(" "" MODELS_IN_STRIP1 "${MODELS_IN}")
string(REGEX REPLACE "," "" MODELS_IN_STRIP2 "${MODELS_IN_STRIP1}")

string(REGEX REPLACE "PARAM_MODEL_IN_REQ\\(" "" MODELS_IN_REQ_STRIP1
    "${MODELS_IN_REQ}")
string(REGEX REPLACE "," "" MODELS_IN_REQ_STRIP2 "${MODELS_IN_REQ_STRIP1}")

string(REGEX REPLACE "PARAM_MODEL_OUT\\(" "" MODELS_OUT_STRIP1 "${MODELS_OUT}")
string(REGEX REPLACE "," "" MODELS_OUT_STRIP2 "${MODELS_OUT_STRIP1}")

set(MODEL_TYPES ${MODELS_IN_STRIP2} ${MODELS_IN_REQ_STRIP2} ${MODELS_OUT_STRIP2})
if (MODEL_TYPES)
  list(REMOVE_DUPLICATES MODEL_TYPES)
endif ()

# Now, generate the definitions of the functions we need.
set(MODEL_PTR_DEFNS "")
set(MODEL_PTR_IMPLS "")
foreach (MODEL_TYPE ${MODEL_TYPES})
  # Generate the definition.
  set(MODEL_PTR_DEFNS "${MODEL_PTR_DEFNS}
// Get the pointer to a ${MODEL_TYPE} parameter.
void* CLI_GetParam${MODEL_TYPE}Ptr(const char* paramName);
// Set the pointer to a ${MODEL_TYPE} parameter.
void CLI_SetParam${MODEL_TYPE}Ptr(const char* paramName, void* ptr);

")

  # Generate the implementation.
  set(MODEL_PTR_IMPLS "${MODEL_PTR_IMPLS}
// Get the pointer to a ${MODEL_TYPE} parameter.
void* CLI_GetParam${MODEL_TYPE}Ptr(const char* paramName)
{
  return (void*) CLI::GetParam<${MODEL_TYPE}*>(paramName);
}

// Set the pointer to a ${MODEL_TYPE} parameter.
void CLI_SetParam${MODEL_TYPE}Ptr(const char* paramName, void* ptr)
{
  CLI::GetParam<${MODEL_TYPE}*>(paramName) = (${MODEL_TYPE}*) ptr;
  CLI::SetPassed(paramName);
}

")
endforeach ()

# Now configure both of the files.
configure_file("${JULIA_H_IN}" "${JULIA_H_OUT}")
configure_file("${JULIA_CPP_IN}" "${JULIA_CPP_OUT}")
