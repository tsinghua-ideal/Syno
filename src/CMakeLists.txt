# Sub-directories
add_subdirectory(Core)

# Library property
set_property(GLOBAL PROPERTY KAS_ALL_LIBS_PROPERTY
        core-lib
        ${Boost_LIBRARIES} -lstdc++ -lm)
get_property(KAS_ALL_LIBS GLOBAL PROPERTY KAS_ALL_LIBS_PROPERTY)
