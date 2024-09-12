class DatasetStat:
    def __init__(self):
        self.total_images = 0
        self.row_counts = {}
        self.col_counts = {}
        self.cell_sizes = []
        self.image_sizes = []
        self.complexities = {}
        self.styles = {}
        self.rotation_angles = []
        self.header_counts = 0
        self.modified_labels_count = 0  # 새로 추가

    def update(self, rows, cols, avg_cell_size, image_size, complexity, style, rotation, with_header, modified_labels):
        self.total_images += 1
        self.row_counts[rows] = self.row_counts.get(rows, 0) + 1
        self.col_counts[cols] = self.col_counts.get(cols, 0) + 1
        self.cell_sizes.append(avg_cell_size)  # avg_cell_size로 변경
        self.image_sizes.append(image_size)
        self.complexities[complexity] = self.complexities.get(complexity, 0) + 1
        self.styles[style] = self.styles.get(style, 0) + 1
        self.rotation_angles.append(rotation)  # rotation으로 변경
        self.modified_labels_count += modified_labels  # 새로 추가
        if with_header:  # with_header로 변경
            self.header_counts += 1

    def get_summary(self):
        return {
            "total_images": self.total_images,
            "row_distribution": self.row_counts,
            "col_distribution": self.col_counts,
            "avg_cell_size": sum(self.cell_sizes) / len(self.cell_sizes) if self.cell_sizes else 0,
            "avg_image_size": sum(self.image_sizes) / len(self.image_sizes) if self.image_sizes else 0,
            "complexity_distribution": self.complexities,
            "style_distribution": self.styles,
            "avg_rotation_angle": sum(self.rotation_angles) / len(self.rotation_angles) if self.rotation_angles else 0,
            "header_percentage": (self.header_counts / self.total_images) * 100 if self.total_images else 0,
            "modified_labels_ratio": (self.modified_labels_count / self.total_images) if self.total_images else 0  # 새로 추가
        }
