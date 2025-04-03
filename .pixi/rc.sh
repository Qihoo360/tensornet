# shellcheck shell=sh

# put activation scripts under .pixi/activate.d/ folder to setup
# development environment, such as export https_proxy.

if [ -d "${PIXI_PROJECT_ROOT:=.}/.pixi/activate.d" ]; then
  __f_err=0
  while IFS='' read -r __f; do
    # shellcheck source=/dev/null
    . "$__f" || __f_err=$?
    if [ $__f_err != 0 ]; then
      if tty -s 2>/dev/null; then
        echo "[WARN] fail to source file $__f with err=$__f_err, skip the following activate scripts." >/dev/tty
      fi
      break
    fi
  done <<-END
	$(find "$PIXI_PROJECT_ROOT/.pixi/activate.d" -name '*.sh')
	END
  ( exit $__f_err )
fi
