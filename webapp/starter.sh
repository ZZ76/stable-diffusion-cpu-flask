export FLASK_APP=stable_diffusion.py
export FLASK_DEBUG=0
while getopts ":d" opt
do
	case $opt in
		d)
			export FLASK_DEBUG=1
	esac
done

if test $FLASK_DEBUG = 1; then
	export USE_MODEL=false
else
	export USE_MODEL=true
fi
flask run --host 0.0.0.0 -p 5000
