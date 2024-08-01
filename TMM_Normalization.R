BiocManager::install("edgeR")
library(rstudioapi)
library(readr)
library(tibble)
library(BiocManager)
library(edgeR)
###################################################################

# Set the path to data directory
#data_path <- "/Users/dr.elsherif/Downloads/NDS2024/data"

# Set working Directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
data_path <- getwd()

# Load count matrix with column names as character
data_exons <- read_csv(file.path(data_path, "m1_patchseq_exon_counts.csv.gz"), col_types = cols())

# Convert the data frame to a matrix, excluding the first column if it contains gene names
count_matrix <- as.matrix(data_exons[,-1])
rownames(count_matrix) <- data_exons[[1]]

# Check for non-numeric values
if (any(!is.numeric(count_matrix))) {
  stop("Count matrix contains non-numeric values")
}

# Create a DGEList object
dge <- DGEList(counts = count_matrix)

# Perform TMM normalization
dge <- calcNormFactors(dge, method = "TMM")

# Extract the normalized counts
normalized_counts <- cpm(dge, normalized.lib.sizes = TRUE)

# Save the normalized counts to a CSV file 
write.csv(normalized_counts, file.path(data_path, "normalized_counts.csv"), row.names = TRUE)

print(head(normalized_counts))

################################## END ############################