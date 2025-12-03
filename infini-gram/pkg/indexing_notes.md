module load GCCcore/13.3.0
module load Rust/1.86.0
cargo build --release
mv target/release/rust_indexing infini_gram/

cd infini_gram/

python -m venv venv
source venv/bin/activate

pip install numpy
pip install tqdm
pip install --upgrade "transformers==4.38.0"




