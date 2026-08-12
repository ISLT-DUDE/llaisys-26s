target("llaisys-device-suda")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    else
        add_cxflags("-wd4819")
    end

    -- Suda device files (pure C++ stubs, no CUDA)
    add_files("../src/device/suda/*.cpp")

    -- Suda operator kernel stubs
    add_files("../src/ops/*/suda/*.cpp")

    on_install(function (target) end)
target_end()
