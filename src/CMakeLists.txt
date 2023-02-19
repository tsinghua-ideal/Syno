# Sub-directories
add_subdirectory(CodeGen)
add_subdirectory(Core)
add_subdirectory(Search)
add_subdirectory(Transforms)

add_library(core-lib STATIC $<TARGET_OBJECTS:core-obj> $<TARGET_OBJECTS:search-obj> $<TARGET_OBJECTS:transforms-obj>)

# Library property
set_property(GLOBAL PROPERTY KAS_ALL_LIBS_PROPERTY
        codegen-lib core-lib
        ${Boost_LIBRARIES} -lstdc++ -lm)
get_property(KAS_ALL_LIBS GLOBAL PROPERTY KAS_ALL_LIBS_PROPERTY)

add_subdirectory(Python)
