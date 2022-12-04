# Sub-directories
add_subdirectory(Core)
add_subdirectory(Search)
add_subdirectory(Transforms)

# Library property
set_property(GLOBAL PROPERTY KAS_ALL_LIBS_PROPERTY
        search-lib transforms-lib core-lib
        ${Boost_LIBRARIES} -lstdc++ -lm)
get_property(KAS_ALL_LIBS GLOBAL PROPERTY KAS_ALL_LIBS_PROPERTY)
